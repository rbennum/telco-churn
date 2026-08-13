"""Paths and constants shared across notebooks.

Import-time side effects are deliberately avoided: no printing, no mkdir. A
module that mutates the filesystem when imported makes notebooks non-reentrant
and surprises anyone who imports it for a single constant.
"""

from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------ constants
RANDOM_SEED = 29
TEST_SIZE = 0.2
TARGET = "Churn"

# ------------------------------------------------------------------ locations
ON_KAGGLE = Path("/kaggle/working").exists()

# Where THIS notebook writes. On Kaggle everything under /kaggle/input is
# read-only, so output always goes to the working directory.
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path("outputs")

# Where PREVIOUS notebooks' artifacts were mounted. Kaggle exposes an attached
# notebook output under /kaggle/input/<something>, and the exact path depends on
# how the input was added, so explicit entries come first and a glob covers the
# rest. Order matters: earlier entries win.
ARTIFACT_DIRS = [
    Path("/kaggle/input/notebooks/benranum/churn-proj-01-ingest"),
    Path("/kaggle/input/churn-proj-01-ingest"),
    OUT_DIR,
    Path("outputs"),
    Path("../outputs"),
    Path("/kaggle/input/churn-proj-02-eda"),
    Path("/kaggle/input/notebooks/benranum/churn-proj-03-baseline"),
]

RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

CANDIDATE_PATHS = [
    Path(f"/kaggle/input/datasets/blastchar/telco-customer-churn/{RAW_FILENAME}"),
    Path(f"/kaggle/input/telco-customer-churn/{RAW_FILENAME}"),
    Path(f"data/raw/{RAW_FILENAME}"),
    Path(f"../data/raw/{RAW_FILENAME}"),
    Path(f"../../data/raw/{RAW_FILENAME}"),
]


# ------------------------------------------------------------------ helpers
def ensure_out_dir() -> Path:
    """Create and return the output directory. Call this explicitly."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def _glob_kaggle_input(filename: str) -> list[Path]:
    """Any matching file anywhere under /kaggle/input, at any nesting depth.

    Survives renaming the upstream notebook, which the hardcoded paths do not.
    """
    root = Path("/kaggle/input")
    return sorted(root.rglob(filename)) if root.exists() else []


def find_artifact(filename: str, extra_dirs: list[Path] | None = None) -> Path:
    """Locate an artifact produced by an earlier notebook.

    Searches ARTIFACT_DIRS in order, then falls back to a recursive glob of
    /kaggle/input. Raises with the full search path on failure, because a
    silent fallback to the wrong file is far more expensive than a crash.
    """
    searched: list[Path] = []
    for d in (extra_dirs or []) + ARTIFACT_DIRS:
        candidate = d / filename
        searched.append(candidate)
        if candidate.exists():
            return candidate

    for hit in _glob_kaggle_input(filename):
        return hit

    raise FileNotFoundError(
        f"Could not find '{filename}'.\n"
        f"Searched:\n" + "\n".join(f"  - {p}" for p in searched) + "\n"
        "On Kaggle: Add Input -> Notebook Output -> select the 01_ingest notebook, "
        "then add its mount path to ARTIFACT_DIRS in telco_churn/config.py.\n"
        "Locally: run notebook 01 first so the file exists in ./outputs."
    )


def find_raw_data() -> Path:
    """Locate the raw Telco CSV."""
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    for hit in _glob_kaggle_input(RAW_FILENAME):
        return hit
    raise FileNotFoundError(
        f"Could not find '{RAW_FILENAME}'.\n"
        f"Searched:\n" + "\n".join(f"  - {p}" for p in CANDIDATE_PATHS) + "\n"
        "On Kaggle: Add Input -> Datasets -> 'Telco Customer Churn' (blastchar).\n"
        "Locally: kaggle datasets download -d blastchar/telco-customer-churn "
        "-p data/raw --unzip"
    )


def describe_paths() -> str:
    """Human-readable path report. Call from a notebook cell, never on import."""
    lines = [f"ON_KAGGLE = {ON_KAGGLE}", f"OUT_DIR   = {OUT_DIR}"]
    for name, fn in [("raw csv", find_raw_data)]:
        try:
            lines.append(f"{name:9} = {fn()}")
        except FileNotFoundError:
            lines.append(f"{name:9} = NOT FOUND")
    for f in ["train.parquet", "test.parquet", "ingest_manifest.json"]:
        try:
            lines.append(f"{f:20} = {find_artifact(f)}")
        except FileNotFoundError:
            lines.append(f"{f:20} = NOT FOUND")
    return "\n".join(lines)
