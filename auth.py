"""PIN storage/verification for Astrid's lock screen. PBKDF2 with a random
salt, stored in a 600-permission file — never plaintext."""
import hashlib
import os
import secrets

ASTRID_HOME = os.path.expanduser("~/.astrid")
PIN_FILE = os.path.join(ASTRID_HOME, "pin")
ITERATIONS = 200_000


def pin_is_set():
    return os.path.exists(PIN_FILE)


def set_pin(pin: str):
    os.makedirs(ASTRID_HOME, exist_ok=True)
    os.chmod(ASTRID_HOME, 0o700)
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, ITERATIONS)
    with open(PIN_FILE, "wb") as f:
        f.write(salt + digest)
    os.chmod(PIN_FILE, 0o600)


def verify_pin(pin: str) -> bool:
    if not pin_is_set():
        return False
    with open(PIN_FILE, "rb") as f:
        data = f.read()
    salt, stored_digest = data[:16], data[16:]
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, ITERATIONS)
    return secrets.compare_digest(digest, stored_digest)
