"""
Vault — Encrypted API Key Storage
===================================
영길님의 Binance API 키를 AES-256-GCM으로 암호화하여 디스크에 저장.

핵심 원칙:
  - 마스터 패스워드는 환경변수(VAULT_PASSPHRASE)로만 입력
  - 평문 키는 메모리에서만 다루고 디스크 저장 금지
  - 키 추가/조회 시점에 감사 로그 남김
  - 잘못된 패스워드 시 명확한 에러, 재시도 제한 없음 (영길님 본인 PC 가정)

설계 결정:
  - PBKDF2-HMAC-SHA256 (480k iterations, OWASP 2023 권장)
  - AES-256-GCM (인증 + 기밀성 동시)
  - Salt + Nonce는 랜덤 생성 후 vault 파일 헤더에 저장
  - cryptography 라이브러리 사용 (PyCA, 표준)

저장 형식 (data/vault.json):
{
  "version": "1.0",
  "kdf": {"algo": "pbkdf2-sha256", "iterations": 480000, "salt": "<base64>"},
  "entries": {
    "binance_testnet": {
      "nonce": "<base64>",
      "ciphertext": "<base64>"      # encrypts {"api_key": "...", "secret": "..."}
    },
    "binance_live": {...}
  }
}
"""
from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console


CONSOLE = Console()


VAULT_VERSION = "1.0"
KDF_ITERATIONS = 480_000   # OWASP 2023 권장
SALT_BYTES = 16
NONCE_BYTES = 12


class VaultError(Exception):
    """Vault 관련 모든 에러"""


class VaultLockedError(VaultError):
    """잘못된 패스워드 또는 손상된 vault"""


@dataclass
class ApiCredential:
    """단일 API 키/시크릿 페어 (메모리에서만 평문)"""
    api_key: str
    secret: str
    label: str = ""             # "binance_testnet" 등
    created_at: str = ""
    permissions: list[str] | None = None    # ["spot", "futures"] etc


def _derive_key(passphrase: str, salt: bytes, iterations: int = KDF_ITERATIONS) -> bytes:
    """PBKDF2로 패스워드 → 32바이트 AES key 유도"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """AES-256-GCM 암호화 → (nonce, ciphertext_with_tag)"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(NONCE_BYTES)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return nonce, ciphertext


def _decrypt(ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:
    """AES-256-GCM 복호화"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except Exception as e:
        raise VaultLockedError(f"Decryption failed: wrong passphrase or corrupted vault ({e})")


class Vault:
    """API 키 보관소"""

    def __init__(self, vault_path: Path | str | None = None):
        self.vault_path = Path(vault_path) if vault_path else (
            Path(__file__).resolve().parent.parent.parent / "data" / "vault.json"
        )
        self._data: dict | None = None

    # =========================================================================
    # File I/O
    # =========================================================================
    def _load_file(self) -> dict:
        if not self.vault_path.exists():
            return {
                "version": VAULT_VERSION,
                "kdf": None,
                "entries": {},
            }
        with open(self.vault_path, "r") as f:
            return json.load(f)

    def _save_file(self, data: dict) -> None:
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vault_path, "w") as f:
            json.dump(data, f, indent=2)
        # Restrict permissions (Unix only)
        try:
            os.chmod(self.vault_path, 0o600)
        except OSError:
            pass    # Windows — NTFS ACL 사용해야 함, 사용자 안내

    def exists(self) -> bool:
        return self.vault_path.exists()

    # =========================================================================
    # Vault Operations
    # =========================================================================
    def _get_passphrase(self) -> str:
        """환경변수에서 패스워드 가져오기"""
        passphrase = os.getenv("VAULT_PASSPHRASE")
        if not passphrase:
            raise VaultError(
                "VAULT_PASSPHRASE environment variable not set. "
                "Set it before using vault: $env:VAULT_PASSPHRASE = 'your_strong_password'"
            )
        if len(passphrase) < 8:
            raise VaultError(
                "VAULT_PASSPHRASE too short (< 8 chars). Use a strong password."
            )
        return passphrase

    def _get_or_create_kdf(self, data: dict) -> tuple[dict, bytes]:
        """KDF salt가 없으면 생성, 있으면 사용 — key 유도까지 수행"""
        passphrase = self._get_passphrase()

        if data["kdf"] is None:
            # First-time setup
            salt = secrets.token_bytes(SALT_BYTES)
            data["kdf"] = {
                "algo": "pbkdf2-sha256",
                "iterations": KDF_ITERATIONS,
                "salt": base64.b64encode(salt).decode("ascii"),
            }
        else:
            salt = base64.b64decode(data["kdf"]["salt"])

        key = _derive_key(passphrase, salt, data["kdf"]["iterations"])
        return data, key

    def add(self, label: str, api_key: str, secret: str,
            permissions: list[str] | None = None,
            overwrite: bool = False) -> None:
        """
        새 자격증명 추가.

        Args:
            label: "binance_testnet", "binance_live" 등 식별자
            api_key: 거래소 API key
            secret: API secret
            permissions: 키 권한 메타데이터 (검증/감사용)
            overwrite: 기존 entry 덮어쓰기 허용
        """
        if not label or not api_key or not secret:
            raise ValueError("label, api_key, secret all required")

        data = self._load_file()
        data, key = self._get_or_create_kdf(data)

        if label in data["entries"] and not overwrite:
            raise VaultError(f"Entry '{label}' already exists. Use overwrite=True.")

        # Plaintext payload
        payload = json.dumps({
            "api_key": api_key,
            "secret": secret,
            "label": label,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "permissions": permissions or [],
        }).encode("utf-8")

        nonce, ciphertext = _encrypt(payload, key)

        data["entries"][label] = {
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

        self._save_file(data)
        CONSOLE.print(f"[green]✓ Vault entry '{label}' saved[/green]")

    def get(self, label: str) -> ApiCredential:
        """라벨로 자격증명 조회 (메모리에서만 평문)"""
        data = self._load_file()

        if label not in data.get("entries", {}):
            raise VaultError(f"Entry '{label}' not found in vault")

        if data["kdf"] is None:
            raise VaultLockedError("Vault has no KDF configured — corrupted")

        passphrase = self._get_passphrase()
        salt = base64.b64decode(data["kdf"]["salt"])
        key = _derive_key(passphrase, salt, data["kdf"]["iterations"])

        entry = data["entries"][label]
        nonce = base64.b64decode(entry["nonce"])
        ciphertext = base64.b64decode(entry["ciphertext"])

        plaintext = _decrypt(ciphertext, key, nonce)
        payload = json.loads(plaintext)

        return ApiCredential(
            api_key=payload["api_key"],
            secret=payload["secret"],
            label=payload.get("label", label),
            created_at=payload.get("created_at", ""),
            permissions=payload.get("permissions", []),
        )

    def list_labels(self) -> list[str]:
        """저장된 라벨 목록 (평문 키 노출 없음)"""
        data = self._load_file()
        return list(data.get("entries", {}).keys())

    def remove(self, label: str) -> bool:
        """자격증명 삭제"""
        data = self._load_file()
        if label not in data.get("entries", {}):
            return False
        del data["entries"][label]
        self._save_file(data)
        CONSOLE.print(f"[yellow]Removed vault entry '{label}'[/yellow]")
        return True


# =============================================================================
# CLI 도움말 — 영길님이 처음 키 등록할 때 사용
# =============================================================================
def setup_cli() -> None:
    """대화형 키 등록"""
    import getpass

    CONSOLE.print("\n[bold cyan]━━━ Vault 초기 설정 ━━━[/bold cyan]\n")

    if not os.getenv("VAULT_PASSPHRASE"):
        CONSOLE.print("[yellow]VAULT_PASSPHRASE 환경변수가 설정되지 않았습니다.[/yellow]")
        CONSOLE.print("PowerShell:")
        CONSOLE.print("  [cyan]$env:VAULT_PASSPHRASE = 'your_strong_password'[/cyan]")
        CONSOLE.print("Bash:")
        CONSOLE.print("  [cyan]export VAULT_PASSPHRASE='your_strong_password'[/cyan]")
        CONSOLE.print()
        CONSOLE.print("[bold red]설정 후 이 스크립트를 다시 실행해주세요.[/bold red]")
        return

    vault = Vault()
    CONSOLE.print(f"Vault path: {vault.vault_path}")

    # 기존 엔트리 표시
    if vault.exists():
        labels = vault.list_labels()
        if labels:
            CONSOLE.print(f"기존 엔트리: {labels}")

    label = input("\nLabel (예: binance_testnet, binance_live): ").strip()
    if not label:
        CONSOLE.print("[red]Cancelled[/red]")
        return

    api_key = getpass.getpass("API Key: ").strip()
    secret = getpass.getpass("API Secret: ").strip()
    perms_str = input("Permissions (콤마 구분, 예: spot,futures,read-only): ").strip()
    permissions = [p.strip() for p in perms_str.split(",")] if perms_str else []

    overwrite = False
    if label in vault.list_labels():
        ans = input(f"'{label}' 이미 존재 — 덮어쓸까요? (y/N): ").strip().lower()
        overwrite = ans == "y"
        if not overwrite:
            CONSOLE.print("[red]Cancelled[/red]")
            return

    vault.add(label, api_key, secret, permissions=permissions, overwrite=overwrite)


if __name__ == "__main__":
    setup_cli()
