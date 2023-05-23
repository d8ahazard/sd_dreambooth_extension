import os
import secrets

from dreambooth import shared

db_path = os.path.join(shared.models_path, "dreambooth")
secret_file = os.path.join(db_path, "secret.txt")
try:
    from core.handlers.config import DirectoryHandler
    dh = DirectoryHandler()
    protected_path = dh.protected_path
    db_path = os.path.join(protected_path, "dreambooth")
    secret_file = os.path.join(db_path, "secret.txt")
except:
    pass
if not os.path.exists(db_path):
    os.makedirs(db_path)


def get_secret():
    secret = ""
    user_key = os.environ.get("API_KEY", None)
    if user_key is not None:
        return user_key
    if not os.path.exists(secret_file):
        return secret
    with open(secret_file, 'r') as file:
        secret = file.read().replace('\n', '')
    return secret


def create_secret():
    secret = str(secrets.token_urlsafe(32))
    with open(secret_file, 'w') as file:
        print(f"Writing new secret {secret} to {secret_file}")
        file.writelines(secret)
    return secret


def clear_secret():
    if os.path.exists(secret_file):
        print(f"Deleting secrets file: {secret_file}")
        os.remove(secret_file)
    return ""


def check_secret(in_secret):
    secret = get_secret()
    return in_secret == secret
