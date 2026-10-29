"""Compute permutation importance, SHAP background and a global SHAP sample (after training)."""
import os
os.environ["TQDM_DISABLE"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.explainability import global_shap, permutation_table
from src.predict import load_bundle

b = load_bundle(); df = pd.read_parquet(C.SNAPSHOTS_PATH)
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
permutation_table(b, va).to_csv(C.METRICS_DIR / "permutation_importance.csv", index=False)
sample, sv, names = global_shap(b, tr, te, n=400)
np.savez_compressed(C.METRICS_DIR / "shap_global.npz", values=sv.values, data=sv.data, names=np.array(names))
print("explainability artefacts written")
