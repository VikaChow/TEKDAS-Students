"""Latih dan evaluasi model klasifikasi status mahasiswa StudentDOA.

Jalankan setelah Data_Prev.py: python Ml_model.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from Data_Prev import TARGET

BASE_DIR = Path(__file__).resolve().parent


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    train_path = BASE_DIR / "student_train.csv"
    test_path = BASE_DIR / "student_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("File train/test belum ada. Jalankan python Data_Prev.py terlebih dahulu.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(str)
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(str)

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y_test.unique())
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": report,
        "test_rows": int(len(y_test)),
    }
    joblib.dump(model, BASE_DIR / "student_outcome_model.pkl")
    (BASE_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    prediction_output = test[[TARGET]].copy()
    prediction_output["Prediksi"] = predictions
    prediction_output.to_csv(BASE_DIR / "student_predictions.csv", index=False)
    print(f"Akurasi test: {metrics['accuracy']:.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("File dibuat: student_outcome_model.pkl, model_metrics.json, student_predictions.csv")


if __name__ == "__main__":
    main()
