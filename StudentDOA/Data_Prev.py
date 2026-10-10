"""Validasi dan siapkan dataset StudentDOA untuk analisis dan machine learning.

Jalankan dari folder ini dengan: python Data_Prev.py
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_StudentDOA.csv"
TARGET = "Target"


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Muat dataset dan pastikan kolom target tersedia."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di {path.name}.")
    if df.empty:
        raise ValueError("Dataset tidak memiliki baris.")
    return df


def prepare_data(df: pd.DataFrame):
    """Pisahkan fitur dan target serta buat data train/test terstratifikasi."""
    data = df.dropna(subset=[TARGET]).copy()
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(str)

    # CSV StudentDOA menggunakan kode numerik untuk kategori dan fitur numerik.
    # Simpan seluruh fitur sebagai numerik agar pipeline ML dapat menangani null.
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    train = X_train.copy()
    train[TARGET] = y_train
    test = X_test.copy()
    test[TARGET] = y_test
    return X, y, train, test


def main():
    df = load_data()
    X, y, train, test = prepare_data(df)
    train.to_csv(BASE_DIR / "student_train.csv", index=False)
    test.to_csv(BASE_DIR / "student_test.csv", index=False)

    report = {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "target": TARGET,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items() if v > 0},
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    (BASE_DIR / "data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Dataset: {len(df):,} baris × {df.shape[1]} kolom")
    print(f"Fitur: {X.shape[1]} | kelas target: {y.nunique()}")
    print("Distribusi target:")
    print(y.value_counts().to_string())
    print(f"Train/test: {len(train):,}/{len(test):,}")
    print("File dibuat: student_train.csv, student_test.csv, data_report.json")


if __name__ == "__main__":
    main()
