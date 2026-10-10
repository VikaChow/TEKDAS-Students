"""Dashboard Streamlit untuk eksplorasi dan prediksi hasil studi StudentDOA.

Urutan menjalankan:
    python Data_Prev.py
    python Ml_model.py
    streamlit run App.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_StudentDOA.csv"
TARGET = "Target"
MODEL_FILE = BASE_DIR / "student_outcome_model.pkl"

st.set_page_config(page_title="Analisis StudentDOA", page_icon="🎓", layout="wide")
st.title("Analisis StudentDOA")
st.caption("Eksplorasi data, kinerja akademik, dan prediksi status akhir mahasiswa")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None


if not DATA_FILE.exists():
    st.error(f"Dataset tidak ditemukan: {DATA_FILE}")
    st.stop()

df = load_data()
df[TARGET] = df[TARGET].astype(str)
model = load_model()
classes = sorted(df[TARGET].dropna().unique())

with st.sidebar:
    st.header("Filter data")
    selected_targets = st.multiselect("Status akhir", classes, default=classes)
    courses = sorted(df["Course"].dropna().unique().tolist())
    selected_courses = st.multiselect("Kode program studi", courses, default=courses)

view = df[df[TARGET].isin(selected_targets) & df["Course"].isin(selected_courses)].copy()
tab_overview, tab_academic, tab_model, tab_data = st.tabs(
    ["Ringkasan", "Kinerja akademik", "Machine Learning", "Jelajahi data"]
)

with tab_overview:
    st.subheader("Gambaran dataset")
    a, b, c, d = st.columns(4)
    a.metric("Mahasiswa sesuai filter", f"{len(view):,}")
    b.metric("Jumlah fitur", f"{df.shape[1] - 1}")
    b.caption("Tidak termasuk kolom Target")
    c.metric("Rata-rata usia masuk", f"{view['Age at enrollment'].mean():.1f} tahun" if len(view) else "-")
    d.metric("Status akhir", f"{view[TARGET].nunique()}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Jumlah mahasiswa per status akhir**")
        if len(view):
            st.bar_chart(view[TARGET].value_counts().rename_axis("Status").to_frame("Jumlah"))
        else:
            st.info("Tidak ada baris yang cocok dengan filter.")
    with right:
        st.markdown("**Status akhir berdasarkan gender**")
        if len(view):
            counts = view.groupby(["Gender", TARGET]).size().unstack(fill_value=0)
            st.bar_chart(counts)
        else:
            st.info("Tidak ada data untuk ditampilkan.")
    st.info("Target menunjukkan status akhir mahasiswa, seperti Dropout, Graduate, atau Enrolled.")

with tab_academic:
    st.subheader("Kinerja semester pertama dan kedua")
    measure = st.radio("Metrik", ["Nilai rata-rata", "Mata kuliah disetujui"], horizontal=True)
    if measure == "Nilai rata-rata":
        cols = ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
    else:
        cols = ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"]
    if len(view):
        summary = view.groupby(TARGET)[cols].mean()
        summary.columns = ["Semester 1", "Semester 2"]
        st.bar_chart(summary)
        st.dataframe(summary.round(2), use_container_width=True)
    else:
        st.info("Pilih setidaknya satu status dan satu program studi.")
    st.caption("Nilai 0 bisa menandakan belum ada evaluasi; periksa konteks sebelum menafsirkannya sebagai nilai akademik nol.")

with tab_model:
    st.subheader("Prediksi status akhir")
    if model is None:
        st.warning("Model belum tersedia. Jalankan `python Data_Prev.py`, lalu `python Ml_model.py`.")
    else:
        row_index = st.selectbox("Pilih nomor baris mahasiswa", df.index.tolist())
        row = df.loc[[row_index]]
        features = row.drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce")
        predicted = model.predict(features)[0]
        st.metric("Prediksi model", str(predicted))
        st.metric("Status aktual", str(row[TARGET].iloc[0]))
        if hasattr(model, "predict_proba"):
            probabilities = pd.Series(model.predict_proba(features)[0], index=model.classes_, name="Probabilitas")
            st.markdown("**Probabilitas per status**")
            st.bar_chart(probabilities)
        metrics_path = BASE_DIR / "model_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.metric("Akurasi data test", f"{metrics['accuracy']:.1%}")

with tab_data:
    st.subheader("Pratinjau baris dan kolom")
    st.write(f"Dataset: {df.shape[0]:,} baris × {df.shape[1]} kolom. Baris setelah filter: {len(view):,}.")
    st.dataframe(view, use_container_width=True, height=480)
    missing = df.isna().sum().rename("Nilai kosong").to_frame()
    missing = missing[missing["Nilai kosong"] > 0]
    if missing.empty:
        st.success("Tidak ditemukan nilai kosong pada dataset.")
    else:
        st.markdown("**Kolom dengan nilai kosong**")
        st.dataframe(missing, use_container_width=True)

st.caption("Sumber data: dataset_StudentDOA.csv")
