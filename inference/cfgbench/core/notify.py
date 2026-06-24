"""Escalation notifications. Telegram via stdlib urllib (no deps); optional SMTP email.

Creds from env; if absent, attempts to source ``~/.config/cfgbench/notify.env`` (simple
``export K=V`` lines). Never raises — a failed notify must not crash a run.

Env:
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID         (telegram backend)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO   (email backend)
    NOTIFY_BACKEND = telegram | email | both     (default: telegram)
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".config" / "cfgbench" / "notify.env"


def _load_env_file() -> None:
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(key: str, default=None):
    if key not in os.environ:
        _load_env_file()
    return os.environ.get(key, default)


def _telegram(text: str) -> bool:
    token, chat = _env("TELEGRAM_BOT_TOKEN"), _env("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, data=data, timeout=10, context=ctx) as r:
            return bool(json.loads(r.read()).get("ok", False))
    except Exception:
        return False


def _email(text: str, subject: str) -> bool:
    host, to = _env("SMTP_HOST"), _env("SMTP_TO")
    if not host or not to:
        return False
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = _env("SMTP_FROM", to)
    msg["To"] = to
    msg.set_content(text)
    try:
        with smtplib.SMTP(host, int(_env("SMTP_PORT", "587")), timeout=15) as s:
            s.starttls()
            user, pw = _env("SMTP_USER"), _env("SMTP_PASS")
            if user and pw:
                s.login(user, pw)
            s.send_message(msg)
        return True
    except Exception:
        return False


def send(text: str, subject: str = "cfgbench alert") -> bool:
    backend = (_env("NOTIFY_BACKEND", "telegram") or "telegram").lower()
    ok = False
    if backend in ("telegram", "both"):
        ok = _telegram(text) or ok
    if backend in ("email", "both"):
        ok = _email(text, subject) or ok
    return ok


def available() -> bool:
    if _env("TELEGRAM_BOT_TOKEN") and _env("TELEGRAM_CHAT_ID"):
        return True
    if _env("SMTP_HOST") and _env("SMTP_TO"):
        return True
    return False
