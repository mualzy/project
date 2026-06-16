from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
log = Path("logs/train_arg_probe.log")
with log.open("w", encoding="utf-8") as f:
    f.write("before import train\n")
    f.flush()
    import train
    f.write("after import train\n")
    f.flush()
    sys.argv = ["train.py", "--help"]
    f.write("before get_args\n")
    f.flush()
    train.get_args()
    f.write("after get_args\n")
    f.flush()
