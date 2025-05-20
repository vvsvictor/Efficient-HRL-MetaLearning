import csv, random, math, numpy as np, torch
from pathlib import Path
from codecarbon import EmissionsTracker

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def csv_append(path: Path, row):
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f); 
        if new: w.writerow(row._fields)  # dataclass/NamedTuple
        w.writerow(row)

def tracker(file: str) -> EmissionsTracker:
    return EmissionsTracker(output_dir="data", output_file=file,
                            allow_multiple_runs=True)
