import importlib
import time
from pathlib import Path

mods = ["torch", "torchvision", "wandb", "evaluate", "unet", "utils.data_loading", "train"]
log = Path("logs/import_probe_root.log")
with log.open("w", encoding="utf-8") as f:
    for mod in mods:
        f.write(f"importing {mod}\n")
        f.flush()
        start = time.time()
        importlib.import_module(mod)
        f.write(f"ok {mod} {time.time() - start:.2f}\n")
        f.flush()
