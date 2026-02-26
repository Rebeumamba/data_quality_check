"""
auth.py — Authentification hardcodée pour Streamlit Cloud
Pas de fichier users.json requis — fonctionne partout.
"""

USERS = {
    "demo": {
        "password": "dataquality2024",
        "role":     "user",
    },
    "admin": {
        "password": "dq-admin-2024",
        "role":     "admin",
    },
}


def verify_login(username: str, password: str):
    user = USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None

def is_logged_in(session_state) -> bool:
    return session_state.get("user") is not None

def get_current_user(session_state):
    return session_state.get("user")

def login(session_state, username: str, password: str) -> bool:
    user = verify_login(username, password)
    if user:
        session_state["user"] = user
        return True
    return False

def logout(session_state):
    session_state["user"] = None
    for key in ["df", "result", "rules", "source_name", "source_type", "step",
                "freshness_h", "alert_t", "detected"]:
        if key in session_state:
            del session_state[key]

def list_users() -> list:
    return [{"username": u, "role": d["role"]} for u, d in USERS.items()]
