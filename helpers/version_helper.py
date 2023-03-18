import json
import os
import subprocess
from typing import Union, Dict

from dreambooth import shared

store_file = os.path.join(shared.dreambooth_models_path, "revision.txt")
change_file = os.path.join(shared.dreambooth_models_path, "changelog.txt")


# Read the current revision from head
def current_revision() -> Union[str, None]:
    if not os.path.exists(os.path.join(shared.extension_path, '.git')):
        return None
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=shared.extension_path, capture_output=True,
                          text=True).stdout.strip()


# Read the stored revision from file
def get_rev() -> Union[str, None]:
    last_rev = None
    if os.path.exists(store_file):
        with open(store_file, "r") as sf:
            last_rev = sf.readlines()[0].strip()
    return last_rev


# Store current revision to file
def store_rev() -> None:
    current = current_revision()
    if current is not None:
        with open(store_file, "w") as sf:
            sf.write(current)


# Store existing changelog for re-retrieval
def store_changes(changes: dict):
    with open(change_file, "w") as cf:
        json.dump(changes, cf)


# Load changes from file
def load_changes():
    changes = None
    if os.path.exists(change_file):
        with open(change_file, "r") as cf:
            changes = json.load(cf)
    return changes


# Check for updates, or return cached changes if force
def check_updates(force: bool = False) -> Union[Dict[str, str], None]:
    last = get_rev()
    current = current_revision()
    if last is None and current:
        last = current
        store_rev()
    changes = {}
    if last and current and last != current:
        changes = get_changes()
        store_changes(changes)
        store_rev()
    elif force:
        changes = load_changes()
    return changes


# Get differences between current and last rev, make changelog
def get_changes() -> Union[Dict[str, str], None]:
    # Check if the shared.extension_path has a .git folder
    if not os.path.exists(os.path.join(shared.extension_path, '.git')):
        return None

    current_branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=shared.extension_path,
                                    capture_output=True, text=True).stdout.strip()

    # Determine the current revision and branch
    current = get_rev()
    if current is None:
        current = current_revision()

    # Get the commit history for the repository and branch
    try:
        commit_history = subprocess.run(
            ['git', 'log', current_branch, '--pretty=format:"%h%x09%an%x09%ad%x09%s"', '--date=format:%Y-%m-%d',
             f"{current}..HEAD"], cwd=shared.extension_path, capture_output=True, text=True).stdout.strip().split(
            '\n')
    except:
        commit_history = []

    # Parse all commits after the current revision
    changes = {}
    for commit in commit_history:
        parts = commit.split('\t')
        if len(parts) < 4:
            continue
        rev = parts[0].lstrip('"')
        author = parts[1]
        date = parts[2]
        title = parts[3].rstrip('"')
        url = f"https://github.com/d8ahazard/sd_dreambooth_extension/commit/{rev}"
        changes[rev] = [title, author, date, url]

    return changes
