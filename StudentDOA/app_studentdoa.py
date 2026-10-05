"""Streamlit dashboard for the StudentDOA student-outcome dataset.

Run from the repository root with:
    streamlit run simulation/app_studentdoa.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "StudentDOA" / "dataset_StudentDOA.csv"
TARGET = "Target"

st.set_page_config(page_title="Analisis StudentDOA", page_icon="🎓", layout="wide")
st.title("Analisis StudentDOA")
st.caption("Dashboard eksplorasi data dan hasil akademik mahasiswa")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df[TARGET] = df[TARGET].astype("category")
    return df


if not DATA_FILE.exists():
    st.error(f"Dataset tidak ditemukan: {DATA_FILE}")
    st.stop()

df = load_data()
classes = sorted(df[TARGET].dropna().astype(str).unique())

with st.sidebar:
    st.header("Filter data")
    selected_targets = st.multiselect("Status akhir", classes, default=classes)
    courses = sorted(df["Course"].dropna().unique().tolist())
    selected_courses = st.multiselect("Kode program studi", courses, default=courses)

view = df[
    df[TARGET].astype(str).isin(selected_targets)
    & df["Course"].isin(selected_courses)
].copy()

tab_overview, tab_academic, tab_data = st.tabs(
    ["Ringkasan", "Kinerja akademik", "Jelajahi data"]
)

with tab_overview:
    st.subheader("Gambaran dataset")
    a, b, c, d = st.columns(4)
    a.metric("Jumlah mahasiswa", f"{len(view):,}")
    b.metric("Jumlah fitur", f"{df.shape[1] - 1}")
    c.metric("Rata-rata usia masuk", f"{view['Age at enrollment'].mean():.1f} tahun" if len(view) else "-")
    d.metric("Status yang dipilih", f"{view[TARGET].nunique()}")

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
            counts = view.groupby(["Gender", TARGET], observed=False).size().unstack(fill_value=0)
            st.bar_chart(counts)

    st.info(
        "Target berisi status akhir mahasiswa, misalnya Dropout, Graduate, atau Enrolled. "
        "Nilai kategori seperti Course dan Application mode disimpan sebagai kode angka."
    )

with tab_academic:
    st.subheader("Kinerja semester pertama dan kedua")
    semester_choice = st.radio(
        "Metrik akademik", ["Nilai rata-rata", "Jumlah mata kuliah disetujui"], horizontal=True
    )
    if semester_choice == "Nilai rata-rata":
        columns = ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
        labels = ["Semester 1", "Semester 2"]
    else:
        columns = ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"]
        labels = ["Semester 1", "Semester 2"]

    if len(view):
        agg = view.groupby(TARGET, observed=False)[columns].mean().rename(
            columns=dict(zip(columns, labels))
        )
        st.bar_chart(agg)
        st.markdown("**Ringkasan akademik per status**")
        st.dataframe(agg.round(2), use_container_width=True)
    else:
        st.info("Pilih setidaknya satu status dan satu program studi.")

    st.caption(
        "Nilai 0 dapat berarti mahasiswa belum memiliki evaluasi/nilai pada semester tersebut; "
        "jangan langsung menganggapnya sebagai nilai akademik nol tanpa memeriksa konteksnya."
    )

with tab_data:
    st.subheader("Pratinjau baris dan kolom")
    st.write(f"Dataset lengkap: {df.shape[0]:,} baris × {df.shape[1]} kolom. Baris setelah filter: {len(view):,}.")
    st.dataframe(view, use_container_width=True, height=480)
    missing = df.isna().sum().rename("Nilai kosong").to_frame()
    missing = missing[missing["Nilai kosong"] > 0]
    if missing.empty:
        st.success("Tidak ditemukan nilai kosong pada dataset.")
    else:
        st.markdown("**Kolom dengan nilai kosong**")
        st.dataframe(missing, use_container_width=True)

st.caption("Sumber data: StudentDOA/dataset_StudentDOA.csv")
