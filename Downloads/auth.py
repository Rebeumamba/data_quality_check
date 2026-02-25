"""
auth.py — Gestion login / password
Stockage local dans users.json (chiffré bcrypt)
Importé par app_v3.py
"""

import json, os, hashlib, hmac, secrets
from datetime import datetime, timedelta
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.json"
SESSION_DURATION_HOURS = 8


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash password avec SHA-256 + sel aléatoire."""
    if salt is None:
        salt = secrets.token_hex(32)
    hashed = hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()
    return hashed, salt

def _verify_password(password: str, hashed: str, salt: str) -> bool:
    h, _ = _hash_password(password, salt)
    return hmac.compare_digest(h, hashed)

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text())
    except:
        return {}

def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


# ══════════════════════════════════════════════════════════════
# API PUBLIQUE
# ══════════════════════════════════════════════════════════════

def create_user(username: str, password: str, role: str = "client") -> bool:
    """
    Crée un nouvel utilisateur.
    role : "admin" ou "client"
    Retourne False si l'username existe déjà.
    """
    users = _load_users()
    if username in users:
        return False
    hashed, salt = _hash_password(password)
    users[username] = {
        "hashed":     hashed,
        "salt":       salt,
        "role":       role,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
    }
    _save_users(users)
    return True

def verify_login(username: str, password: str) -> dict | None:
    """
    Vérifie les credentials.
    Retourne le profil utilisateur ou None si invalide.
    """
    users = _load_users()
    if username not in users:
        return None
    user = users[username]
    if not _verify_password(password, user["hashed"], user["salt"]):
        return None
    # Mettre à jour last_login
    users[username]["last_login"] = datetime.now().isoformat()
    _save_users(users)
    return {"username": username, "role": user["role"]}

def user_exists(username: str) -> bool:
    return username in _load_users()

def list_users() -> list:
    users = _load_users()
    return [
        {"username": u, "role": d["role"], "last_login": d.get("last_login")}
        for u, d in users.items()
    ]

def delete_user(username: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    del users[username]
    _save_users(users)
    return True

def change_password(username: str, new_password: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    hashed, salt = _hash_password(new_password)
    users[username]["hashed"] = hashed
    users[username]["salt"]   = salt
    _save_users(users)
    return True


# ══════════════════════════════════════════════════════════════
# INIT — Créer un admin par défaut si aucun user
# ══════════════════════════════════════════════════════════════

def init_default_admin():
    """
    Crée l'admin par défaut au premier lancement.
    Modifier le mot de passe immédiatement après !
    """
    if not _load_users():
        create_user("admin", "admin1234", role="admin")
        print("⚠️  Admin par défaut créé : admin / admin1234")
        print("   Changez le mot de passe immédiatement !")

init_default_admin()


# ══════════════════════════════════════════════════════════════
# STREAMLIT SESSION HELPERS
# ══════════════════════════════════════════════════════════════

def is_logged_in(session_state) -> bool:
    return session_state.get("user") is not None

def get_current_user(session_state) -> dict | None:
    return session_state.get("user")

def login(session_state, username: str, password: str) -> bool:
    user = verify_login(username, password)
    if user:
        session_state["user"] = user
        return True
    return False

def logout(session_state):
    session_state["user"] = None
    # Nettoyer toutes les données de session
    for key in ["df","result","rules","source_name","source_type","step",
                "freshness_h","alert_t","detected"]:
        if key in session_state:
            del session_state[key]
