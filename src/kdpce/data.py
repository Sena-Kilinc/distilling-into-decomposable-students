"""Dataset loading and leakage-safe splitting.

Raw files are read from datasets/raw/ (official UCI zips, see datasets/README.md).
All splits are made BEFORE any preprocessing; every data-driven transform
(PCE bases, one-hot levels, teacher, students) is fitted on the training part only.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[2]
import os
RAW = Path(os.environ.get("KDPCE_RAW", ROOT / "datasets" / "raw"))


@dataclass
class Dataset:
    name: str
    X: dict            # split -> DataFrame (raw predictors)
    y: dict            # split -> np.ndarray of int labels
    feature_names: list
    feature_kinds: list   # "numeric" | "categorical"
    class_names: list
    groups: dict | None   # feature -> group name (for group-level comparisons)
    info: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find(patterns):
    for p in patterns:
        hits = sorted(RAW.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"None of {patterns} found in {RAW}. See datasets/README.md")


# --------------------------------------------------------------------------
# UCI Online Shoppers Purchasing Intention (id 468)
# --------------------------------------------------------------------------
OSPI_CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region",
                    "TrafficType", "VisitorType", "Weekend"]


def load_ospi(seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2) -> Dataset:
    src = _find(["online_shoppers_intention.csv", "*online*shoppers*.zip", "*468*.zip"])
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(io.BytesIO(z.read(name)))
    else:
        df = pd.read_csv(src)
    n_raw = len(df)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    y = df.pop("Revenue").astype(int).to_numpy()
    for c in OSPI_CATEGORICAL:
        df[c] = df[c].astype(str)
    feats = list(df.columns)
    kinds = ["categorical" if c in OSPI_CATEGORICAL else "numeric" for c in feats]
    idx = np.arange(len(df))
    i_trv, i_te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    i_tr, i_va = train_test_split(i_trv, test_size=val_frac / (1 - test_frac),
                                  stratify=y[i_trv], random_state=seed)
    X = {s: df.iloc[i].reset_index(drop=True) for s, i in
         [("train", i_tr), ("val", i_va), ("test", i_te)]}
    Y = {"train": y[i_tr], "val": y[i_va], "test": y[i_te]}
    info = dict(source=str(src.name), sha256=_sha256(src), n_raw=n_raw,
                n_exact_duplicates_removed=n_dup, n_after_dedup=len(df),
                n_predictors=len(feats), positive_rate=float(y.mean()),
                split="stratified 60/20/20", seed=seed,
                sizes={k: int(len(v)) for k, v in Y.items()})
    return Dataset("ospi", X, Y, feats, kinds, ["no_purchase", "purchase"], None, info)


# --------------------------------------------------------------------------
# UCI HAR (id 240): official subject-wise train/test split
# --------------------------------------------------------------------------
def _har_group(name: str) -> str:
    # e.g. '12 tBodyAcc-mean()-X' -> 'tBodyAcc'; 'angle(tBodyAccMean,gravity)' -> 'angle'
    base = name.split(" ", 1)[1] if " " in name else name
    if base.startswith("angle"):
        return "angle"
    return base.split("-")[0]


def load_har(seed: int = 0, n_val_subjects: int = 5) -> Dataset:
    src = _find(["*human*activity*.zip", "*240*.zip", "UCI HAR Dataset.zip", "UCI HAR Dataset"])

    def read_from(get):
        feats = [l.split() for l in get("features.txt").decode().strip().splitlines()]
        names = [f"{int(i):03d} {n}" for i, n in feats]  # unique (raw names contain duplicates)
        labels = [l.split()[1] for l in get("activity_labels.txt").decode().strip().splitlines()]
        out = {}
        for part in ["train", "test"]:
            Xp = np.loadtxt(io.BytesIO(get(f"{part}/X_{part}.txt")))
            yp = np.loadtxt(io.BytesIO(get(f"{part}/y_{part}.txt")), dtype=int) - 1
            sp = np.loadtxt(io.BytesIO(get(f"{part}/subject_{part}.txt")), dtype=int)
            out[part] = (pd.DataFrame(Xp, columns=names), yp, sp)
        return names, labels, out

    if src.is_dir():
        base = src
        names, labels, out = read_from(lambda p: (base / p).read_bytes())
    else:
        z = zipfile.ZipFile(src)
        inner = [n for n in z.namelist() if n.endswith(".zip")]
        if inner:  # UCI wraps 'UCI HAR Dataset.zip' inside the download
            z = zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
        prefix = [n for n in z.namelist() if n.endswith("features.txt")][0][: -len("features.txt")]
        names, labels, out = read_from(lambda p: z.read(prefix + p))

    Xtr_all, ytr_all, str_all = out["train"]
    Xte, yte, ste = out["test"]
    gss = GroupShuffleSplit(n_splits=1, test_size=n_val_subjects, random_state=seed)
    i_tr, i_va = next(gss.split(Xtr_all, ytr_all, groups=str_all))
    X = {"train": Xtr_all.iloc[i_tr].reset_index(drop=True),
         "val": Xtr_all.iloc[i_va].reset_index(drop=True),
         "test": Xte.reset_index(drop=True)}
    Y = {"train": ytr_all[i_tr], "val": ytr_all[i_va], "test": yte}
    S = {"train": str_all[i_tr], "val": str_all[i_va], "test": ste}
    assert not (set(S["train"]) & set(S["test"])) and not (set(S["val"]) & set(S["test"]))
    assert not (set(S["train"]) & set(S["val"]))
    info = dict(source=str(src.name), sha256=_sha256(src) if src.is_file() else None,
                n_total=int(len(Xtr_all) + len(Xte)), n_predictors=len(names),
                split=f"official subject split; {n_val_subjects} train subjects held out for validation",
                subjects={k: sorted(map(int, set(v))) for k, v in S.items()},
                sizes={k: int(len(v)) for k, v in Y.items()}, seed=seed,
                class_counts={k: np.bincount(v, minlength=6).tolist() for k, v in Y.items()})
    groups = {n: _har_group(n) for n in names}
    return Dataset("har", X, Y, names, ["numeric"] * len(names), labels, groups, info)


def load(name: str, seed: int = 0) -> Dataset:
    return {"ospi": load_ospi, "har": load_har}[name](seed=seed)


def onehot(Xdf: dict, feats, kinds):
    """One-hot for CART (fitted on train levels only). Returns arrays + column->feature map."""
    cats = [f for f, k in zip(feats, kinds) if k == "categorical"]
    levels = {c: sorted(Xdf["train"][c].unique()) for c in cats}
    cols, owner = [], []
    for f, k in zip(feats, kinds):
        if k == "categorical":
            for lv in levels[f]:
                cols.append((f, lv)); owner.append(f)
        else:
            cols.append((f, None)); owner.append(f)
    out = {}
    for s, df in Xdf.items():
        mats = []
        for f, lv in cols:
            mats.append((df[f] == lv).to_numpy(float) if lv is not None else df[f].to_numpy(float))
        out[s] = np.column_stack(mats)
    names = [f if lv is None else f"{f}={lv}" for f, lv in cols]
    return out, names, owner
