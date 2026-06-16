import importlib
import time
from pathlib import Path

log = Path("logs/import_probe.log")
mods = [
    "argparse",
    "logging",
    "os",
    "random",
    "sys",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "pathlib",
    "torch.optim",
    "torch.utils.data",
    "tqdm",
    "wandb",
    "evaluate",
    "unet",
    "utils.data_loading",
    "utils.dice_score",
]

log.parent.mkdir(exist_ok=True)
with log.open("w", encoding="utf-8") as f:
    for mod in mods:
        f.write(f"importing {mod}\n")
        f.flush()
        start = time.time()
        importlib.import_module(mod)
        f.write(f"ok {mod} {time.time() - start:.2f}\n")
        f.flush()
