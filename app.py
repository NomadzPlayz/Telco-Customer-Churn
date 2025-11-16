# app.py
"""
Streamlit app: Telco Customer Churn Predictor (robust version)

Fitur utama:
- Load model artifact (joblib dict: {'model', 'scaler', 'columns', 'numeric_cols'})
- Jika artifact tidak ada: opsional jalankan training (src/train.py) otomatis
- Validasi input single-sample form & batch CSV upload
- Auto-align fitur (menambahkan kolom yang hilang dengan 0, re-order kolom)
- Prediksi probabilitas + kelas, tampilkan table hasil (batch) atau metric untuk single
- SHAP explainability (try/except, non-blocking)
- Logging sederhana ke Streamlit (ui)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/lgbm_churn.joblib"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAIN_ON_MISSING = True

st.set_page_config(page_title="Churn Predictor (Robust)", layout="wide")

def run_training_script():
    train_script = os.path.join("src", "train.py")
    if not os.path.exists(train_script):
        st.error("Training script tidak ditemukan di src/train.py. Tidak bisa melakukan fallback training.")
        return False
    cmd = ["python", train_script]
    st.info("Model artifact tidak ditemukan. Menjalankan training fallback (development).")
    with st.spinner("Training sedang berjalan — cek logs di bawah."):
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = process.stdout.readline()
                if line:
                    st.text(line.strip())
                elif process.poll() is not None:
                    break
            rc = process.returncode
            if rc != 0:
                st.error(f"Training script keluar dengan kode {rc}. Lihat logs di atas.")
                return False
            st.success("Training selesai — artifact harusnya tersedia sekarang.")
            return True
        except Exception as e:
            st.exception(e)
            return False

@st.cache_resource
def load_artifact(path=MODEL_PATH):
    if not os.path.exists(path):
        st.warning(f"Model artifact tidak ditemukan: {path}")
        if TRAIN_ON_MISSING:
            ok = run_training_script()
            if not ok:
                return None
        else:
            return None
    try:
        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            artifact.setdefault('scaler', None)
            artifact.setdefault('columns', None)
            artifact.setdefault('numeric_cols', [])
            return artifact
        else:
            return {'model': artifact, 'scaler': None, 'columns': None, 'numeric_cols': []}
    except Exception as e:
        st.error(f"Gagal load artifact: {e}")
        return None

def validate_and_prepare_input(df_input: pd.DataFrame, artifact: dict):
    if artifact is None:
        raise ValueError("Artifact model tidak tersedia.")
    cols = artifact.get('columns')
    scaler = artifact.get('scaler')
    numeric_cols = artifact.get('numeric_cols', [])
    if cols is None:
        try:
            cols = artifact['model'].feature_name()
        except Exception:
            cols = df_input.columns.tolist()
    df = df_input.copy().reset_index(drop=True)
    obj_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in obj_cols:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            dummies = pd.get_dummies(df[c], prefix=c)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
    aligned = pd.DataFrame(columns=cols, index=df.index)
    for c in cols:
        if c in df.columns:
            aligned[c] = df[c]
        else:
            aligned[c] = 0
    extra = [c for c in df.columns if c not in cols]
    if extra:
        st.info(f"Ada kolom input yang tidak dipakai (diabaikan): {', '.join(extra[:10])}" + (" ..." if len(extra)>10 else ""))
    aligned = aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
    if scaler is not None and hasattr(scaler, "mean_") and len(numeric_cols)>0:
        existing_num = [c for c in numeric_cols if c in aligned.columns]
        if existing_num:
            try:
                aligned[existing_num] = scaler.transform(aligned[existing_num])
            except Exception as e:
                st.warning(f"Gagal scaling numeric cols: {e}")
    return aligned

def predict_df(X: pd.DataFrame, artifact: dict):
    model = artifact['model']
    try:
        proba = model.predict(X)
    except Exception:
        proba = model.predict(X.values)
    preds = (proba >= 0.5).astype(int)
    return pd.DataFrame({
        "pred": preds,
        "probability_churn": proba
    }, index=X.index)

st.title("Telco Churn Predictor — Robust Deployment Demo")
st.write("Aplikasi demo: auto-align fitur, batch CSV, dan fallback training jika artifact hilang.")

artifact = load_artifact()
if artifact is None:
    st.error("Model artifact tidak tersedia dan fallback training gagal atau dimatikan. Upload model ke models/ atau jalankan src/train.py.")
    st.stop()

st.sidebar.header("Input")
batch_file = st.sidebar.file_uploader("Upload CSV untuk prediksi batch (opsional)", type=['csv'])
use_example = st.sidebar.checkbox("Gunakan contoh dataset (data/ file) untuk demo", value=False)

if use_example:
    if os.path.exists(DATA_PATH):
        df_example = pd.read_csv(DATA_PATH).head(50)
        st.sidebar.write("Contoh dataset ter-load (50 baris pertama).")
        st.sidebar.write(df_example.head(3))
        batch_df = df_example.copy()
    else:
        st.sidebar.warning(f"Contoh data tidak ditemukan: {DATA_PATH}")
        batch_df = None
else:
    batch_df = None

if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)
        st.sidebar.success(f"File ter-upload: {batch_file.name} (baris={len(batch_df)})")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca CSV: {e}")
        batch_df = None

st.sidebar.subheader("Atau: isi manual (single)")
form = st.sidebar.form(key="single_form")
gender = form.selectbox("Gender", ["Female","Male"])
senior = form.selectbox("SeniorCitizen", [0,1], index=0)
partner = form.selectbox("Partner", ["No","Yes"])
dependents = form.selectbox("Dependents", ["No","Yes"])
tenure = form.number_input("tenure (months)", min_value=0, max_value=200, value=12)
phone_service = form.selectbox("PhoneService", ["No","Yes"])
monthly = form.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0)
total = form.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=monthly*tenure)
submitted = form.form_submit_button("Gunakan input manual")

# Persist manual single sample to session_state so it survives reruns (e.g. when clicking Predict)
if submitted:
    sample = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])
    st.session_state["single_df"] = sample
    st.sidebar.success("Single sample siap diprediksi (periksa di main panel).")

# If a batch file was uploaded, remove any previous single_df to avoid confusion
if batch_df is not None and "single_df" in st.session_state:
    try:
        del st.session_state["single_df"]
    except Exception:
        pass

# Read single_df from session_state (if exists)
single_df = st.session_state.get("single_df", None)

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Preview input")
    if batch_df is not None:
        st.write("Batch CSV preview:")
        st.dataframe(batch_df.head(10))
    elif single_df is not None:
        st.write("Single sample preview:")
        st.dataframe(single_df)
    else:
        st.info("Belum ada input. Upload CSV atau isi form di sidebar.")

    if st.button("Predict"):
        input_df = None
        is_batch = False
        if batch_df is not None:
            input_df = batch_df.copy()
            is_batch = True
        elif single_df is not None:
            input_df = single_df.copy()
            is_batch = False
        else:
            st.warning("Tidak ada data untuk diprediksi.")
            st.stop()

        try:
            with st.spinner("Menyiapkan fitur dan membuat prediksi..."):
                X_ready = validate_and_prepare_input(input_df, artifact)
                results = predict_df(X_ready, artifact)
                out = pd.concat([input_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
                out['pred_label'] = out['pred'].map({1:"Yes", 0:"No"})
                st.success("Prediksi selesai.")
                st.dataframe(out.head(100))
                if is_batch:
                    st.write("Ringkasan probabilitas churn:")
                    st.write(out['probability_churn'].describe().to_frame().T)
                else:
                    p = float(out.at[0,'probability_churn'])
                    st.metric(label="Predicted churn (Yes/No)", value=out.at[0,'pred_label'])
                    st.write(f"Probability churn: {p:.4f}")
        except Exception as e:
            st.exception(e)

        if not is_batch:
            try:
                if st.checkbox("Tampilkan SHAP explanation (single sample)"):
                    st.info("Menggenerate SHAP values (mungkin lambat pada instance tanpa wheel shap).")
                    import shap
                    model = artifact['model']
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_ready)
                    st.write("SHAP values (waterfall):")
                    fig = shap.plots.waterfall(shap_values=shap_vals[0], show=False)
                    st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.warning(f"SHAP failed: {e}")

with col2:
    st.subheader("Model & artifact info")
    try:
        st.write("Artifact keys present:")
        st.write(list(artifact.keys()))
        st.write("Saved feature count:", len(artifact.get('columns') or []))
        st.write("Numeric cols (from artifact):", artifact.get('numeric_cols', []))
        metrics_path = os.path.join("models", "metrics.json")
        if os.path.exists(metrics_path):
            try:
                st.write("Training metrics (models/metrics.json):")
                st.json(eval(open(metrics_path,'r').read()))
            except Exception:
                st.write("Tidak bisa load metrics.json cleanly.")
    except Exception as e:
        st.write("Gagal menampilkan artifact info:", e)

st.write("---")
st.write("Notes:")
st.write("""
- Aplikasi ini **mencoba** fallback ke training jika model artifact tidak ditemukan. Di lingkungan produksi, jangan lakukan ini — pretrain dan simpan artifact di storage/push ke repo.
- Untuk hasil terbaik: simpan model dengan struktur:
  joblib.dump({
    'model': model,
    'scaler': scaler,
    'columns': X_train.columns.tolist(),
    'numeric_cols': ['tenure','MonthlyCharges','TotalCharges', ...]
  }, 'models/lgbm_churn.joblib')
- Jika Anda ingin REST API, buat FastAPI terpisah yang memanggil fungsi validate_and_prepare_input().
""")

