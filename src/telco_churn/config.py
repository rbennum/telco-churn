from pathlib import Path

RANDOM_SEED = 29
TEST_SIZE = 0.2
TARGET = "Churn"

ON_KAGGLE = Path("/kaggle/working").exists()
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path("outputs")

CANDIDATE_PATHS = [
    Path(
        "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ),
    Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    Path("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    Path("../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
]
