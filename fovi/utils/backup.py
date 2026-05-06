import os
import shutil
from ..paths import SAVE_DIR, SLOW_DIR

def backup_model(base_fn):
    shutil.copytree(f"{SAVE_DIR}/logs/{base_fn}", f"{SLOW_DIR}/logs/{base_fn}", dirs_exist_ok=True)

def backup_models(base_fns):
    for base_fn in base_fns:
        backup_model(base_fn)