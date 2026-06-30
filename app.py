import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ====================================================
# KONFIGURASI HALAMAN
# ====================================================

st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("Prediksi Risiko Diabetes")
st.write("Implementasi Algoritma Decision Tree Berdasarkan Dataset Medis")

# ====================================================
# MEMBACA DATASET
# ====================================================

try:
    data = pd.read_csv("diabetes.csv")
except Exception as e:
    st.error(f"Gagal membaca dataset : {e}")
    st.stop()

# ====================================================
# MENAMPILKAN DATASET AWAL
# ====================================================

st.header("1. Dataset Awal")
st.write(data.head())
st.write("Jumlah Data :", data.shape[0])
st.write("Jumlah Kolom :", data.shape[1])

# ====================================================
# CEK MISSING VALUE
# ====================================================

st.header("2. Missing Value")
missing = data.isnull().sum()
st.write(missing)

# ====================================================
# MENGHAPUS DATA KOSONG
# ====================================================

data = data.dropna()
st.success(f"Jumlah data setelah preprocessing : {len(data)}")

# ====================================================
# MEMISAHKAN FITUR DAN TARGET
# ====================================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

st.success("Data berhasil dipisahkan menjadi fitur dan target.")

# ====================================================
# SPLIT DATA
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.write("Jumlah Data Training :", len(X_train))
st.write("Jumlah Data Testing :", len(X_test))

# ====================================================
# MEMBANGUN MODEL DECISION TREE
# ====================================================

st.header("3. Model Decision Tree")

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)

st.success("Model Decision Tree berhasil dibuat.")

# ====================================================
# PREDIKSI
# ====================================================

y_pred = model.predict(X_test)

# ====================================================
# AKURASI
# ====================================================

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Accuracy")
st.metric(
    label="Nilai Accuracy",
    value=f"{accuracy:.2%}"
)

# ====================================================
# CLASSIFICATION REPORT
# ====================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# tampilkan tabel lebih kecil
st.table(report_df.round(2))

# ====================================================
# CONFUSION MATRIX
# ====================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4,4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tidak Diabetes", "Diabetes"]
)
disp.plot(ax=ax, cmap="Blues")
st.pyplot(fig)

# ====================================================
# FEATURE IMPORTANCE
# ====================================================

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.table(importance.round(4))

fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.barh(importance["Fitur"], importance["Importance"])
ax2.set_xlabel("Importance")
ax2.set_ylabel("Fitur")
ax2.set_title("Feature Importance Decision Tree")
plt.tight_layout()
st.pyplot(fig2)

# ====================================================
# VISUALISASI DECISION TREE
# ====================================================

st.header("4. Visualisasi Decision Tree")

fig3, ax3 = plt.subplots(figsize=(18,8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Tidak Diabetes", "Diabetes"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,
    ax=ax3
)
st.pyplot(fig3)

# ====================================================
# INPUT MANUAL UNTUK PREDIKSI
# ====================================================

st.header("5. Prediksi Pasien Baru")

with st.form("input_pasien"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submit = st.form_submit_button("Prediksi")

if submit:
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=X.columns)

    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("Hasil Prediksi: Pasien berisiko Diabetes")
    else:
        st.success("Hasil Prediksi: Pasien tidak berisiko Diabetes")
