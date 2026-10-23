"""
train.py — Training pipeline runner inside src/models.
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def run_training():
    import subprocess
    return subprocess.run([sys.executable, "train.py"], cwd=str(root_dir))

if __name__ == "__main__":
    run_training()
