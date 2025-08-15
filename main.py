import os
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import google.generativeai as genai
import cv2
from deepface import DeepFace
from PIL import Image
import time
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun
# ===================== Page Config & Styles =====================
st.set_page_config(page_title="Advanced Data Profiling + Face Auth",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-container { background:#f6f7fb; padding:10px; border-radius:12px; }
.small { color:#6b7280; font-size:12px; }
.auth-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 2rem 0;
}
.success-auth {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;
}
.failed-auth {
    background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ===================== Session State (single-shot init) =====================
if "initialized" not in st.session_state:
    st.session_state.update({
        "processed_data": None,
        "original_data": None,
        "last_uploaded_file": None,
        "gemini_chat": None,
        "gemini_key": None,
        "data_brief": "",
        "authenticated": False,
        "user_face_encoding": None,
        "auth_attempts": 0,
        "max_auth_attempts": 3,
        "registered_users": {},      # user_id -> embedding
        "current_user": None,

        # registration flow flags
        "registration_complete": False,
        "just_registered_user": None
    })
    st.session_state.initialized = True

# ===================== Face Auth Helpers =====================
class DeepFaceTransformer(VideoTransformerBase):
    def __init__(self, model_name="Facenet"):
        self.face_embedding = None
        self.error = None
        self.model_name = model_name

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.extract_faces(
                rgb_frame, detector_backend="opencv", enforce_detection=False
            )
            if not result or len(result) == 0 or ("face" not in result[0]) or result[0]["face"] is None:
                self.error = "No face detected in the webcam frame"
                return img
            if len(result) > 1:
                self.error = "Multiple faces detected; show only one face"
                return img

            face = result[0]["face"]
            face_uint8 = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
            embedding = DeepFace.represent(
                face_uint8, model_name=self.model_name, detector_backend="skip"
            )[0]["embedding"]

            self.face_embedding = embedding
            self.error = None
        except Exception as e:
            self.error = f"Error processing webcam frame: {str(e)}"

        return img


def encode_face_from_image(image, model_name="Facenet", debug=False):
    try:
        img_array = np.array(image)
        # Normalize to RGB (drop alpha if present)
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        if img_array.ndim == 2:  # grayscale to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        rgb_image = img_array if img_array.shape[-1] == 3 else cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        if debug:
            st.write(f"Debug: Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

        result = DeepFace.extract_faces(rgb_image, detector_backend="opencv", enforce_detection=False)

        if debug:
            st.write(f"Debug: DeepFace result: {len(result)} faces")

        if not result or len(result) == 0:
            return None, "No face detected in the image."
        if len(result) > 1:
            return None, "Multiple faces detected. Please upload a single-face image."

        face = result[0]["face"]
        if face is None or face.size == 0:
            return None, "Invalid face array."

        if debug:
            st.write(f"Debug: Face shape: {face.shape}, max: {face.max()}, dtype: {face.dtype}")

        face_uint8 = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
        embedding = DeepFace.represent(face_uint8, model_name=model_name, detector_backend="skip")[0]["embedding"]
        return embedding, None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def authenticate_face(test_embedding, registered_users, threshold=10.0):
    for user_id, ref_embedding in registered_users.items():
        try:
            distance = np.linalg.norm(np.array(test_embedding) - np.array(ref_embedding))
            if distance < threshold:
                return True, user_id
        except Exception as e:
            st.error(f"Verification error for {user_id}: {str(e)}")
    return False, None


def register_user_face():
    st.markdown('<div class="auth-container"><h3>👤 Register New User</h3></div>', unsafe_allow_html=True)

    # Agar register ho chuka hai to success screen dikhao
    if st.session_state.get("registration_complete", False):
        u = st.session_state.get("just_registered_user", "")
        st.success(f'🎉 User "{u}" registered successfully!')
        st.caption("You’re all set. Continue to the app or register another user.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Continue to App (Auto-Login)", key="reg_go_to_app"):
                st.session_state.authenticated = True
                st.session_state.current_user = u
                st.session_state.show_register = False  # Registration page se nikalne ka flag
                st.rerun()
        with col2:
            if st.button("↩️ Register Another User", key="reg_another_user"):
                st.session_state.registration_complete = False
                st.session_state.just_registered_user = None
                st.session_state.register_user_id = ""
                st.session_state.register_upload = None
                st.rerun()
        st.stop()

    # Normal registration form
    st.text_input("Enter User ID/Name:", key="register_user_id")

    option = st.radio("Face Recognition Model", ["Facenet"], key="register_model")

    st.subheader("📷 Option 1: Upload Photo")
    st.file_uploader(
        "Upload a clear photo with one face",
        type=["jpg", "jpeg", "png"],
        key="register_upload"
    )

    st.subheader("📹 Option 2: Webcam Capture")
    from streamlit_webrtc import webrtc_streamer
    ctx = webrtc_streamer(
        key=f"webcam_register_{st.session_state.get('register_user_id', 'new')}",
        video_transformer_factory=None,
        media_stream_constraints={"video": True, "audio": False}
    )

    if st.button("✅ Done", key="register_done"):
        # Save registration state
        st.session_state.registration_complete = True
        st.session_state.just_registered_user = st.session_state.get("register_user_id", "")
        st.rerun()


def face_login():
    st.markdown('<div class="auth-container"><h3>🔐 Face Authentication Login</h3></div>', unsafe_allow_html=True)

    if len(st.session_state.registered_users) == 0:
        st.warning("⚠️ No registered users found. Please register first.")
        return False

    st.info(f"👥 Registered Users: {', '.join(st.session_state.registered_users.keys())}")

    model_name = st.selectbox("Face Recognition Model",
                              ["Facenet", "VGG-Face", "DeepFace", "ArcFace"],
                              key="auth_model")
    threshold = st.slider("Face Match Threshold (lower = stricter)", 5.0, 20.0, 10.0, 0.5)

    st.subheader("📷 Upload Photo to Authenticate")
    auth_image = st.file_uploader("Upload a clear photo with one face",
                                  type=['jpg', 'jpeg', 'png'],
                                  key="auth_upload")

    if auth_image is not None:
        image = Image.open(auth_image).convert("RGB")
        st.image(image, caption="Authentication Photo", width=300)

        if st.button("Authenticate with Photo", key="auth_btn"):
            with st.spinner("Verifying identity..."):
                embedding, error = encode_face_from_image(image, model_name)
                if embedding is not None:
                    is_match, matched_user = authenticate_face(embedding, st.session_state.registered_users, threshold)
                    if is_match:
                        st.session_state.authenticated = True
                        st.session_state.current_user = matched_user
                        st.session_state.auth_attempts = 0
                        st.markdown(f'<div class="success-auth">✅ Welcome back, {matched_user}!</div>', unsafe_allow_html=True)
                        st.balloons()
                        time.sleep(0.6)
                        st.rerun()
                        return True
                    else:
                        st.session_state.auth_attempts += 1
                        remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                        st.markdown(f'<div class="failed-auth">❌ Authentication failed. {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                        st.info("💡 Try a clearer, well-lit photo.")
                else:
                    st.markdown(f'<div class="failed-auth">❌ Authentication error: {error}</div>', unsafe_allow_html=True)
                    st.info("💡 Ensure one clear face is visible.")

    st.subheader("📹 Webcam Authentication")
    ctx = webrtc_streamer(
        key="auth_webcam",
        video_transformer_factory=lambda: DeepFaceTransformer(model_name),
        media_stream_constraints={"video": True, "audio": False}
    )

    if ctx and ctx.video_transformer:
        if st.button("Authenticate with Webcam", key="webcam_auth_btn"):
            with st.spinner("Verifying identity..."):
                transformer = ctx.video_transformer
                if transformer.error:
                    st.markdown(f'<div class="failed-auth">❌ Authentication failed: {transformer.error}</div>', unsafe_allow_html=True)
                    st.info("💡 Make sure only your face is visible and lighting is good.")
                elif transformer.face_embedding is not None:
                    is_match, matched_user = authenticate_face(transformer.face_embedding, st.session_state.registered_users, threshold)
                    if is_match:
                        st.session_state.authenticated = True
                        st.session_state.current_user = matched_user
                        st.session_state.auth_attempts = 0
                        st.markdown(f'<div class="success-auth">✅ Welcome back, {matched_user}!</div>', unsafe_allow_html=True)
                        st.balloons()
                        time.sleep(0.6)
                        st.rerun()
                        return True
                    else:
                        st.session_state.auth_attempts += 1
                        remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                        st.markdown(f'<div class="failed-auth">❌ Authentication failed. {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                        st.info("💡 Adjust your position or lighting.")
                else:
                    st.markdown('<div class="failed-auth">❌ Could not process webcam frame.</div>', unsafe_allow_html=True)

    return False


def show_auth_status():
    if st.session_state.authenticated:
        st.sidebar.success(f"🟢 Authenticated as: {st.session_state.current_user}")
        if st.sidebar.button("🚪 Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
    else:
        st.sidebar.error("🔴 Not Authenticated")
        st.sidebar.info(f"Auth attempts: {st.session_state.auth_attempts}/{st.session_state.max_auth_attempts}")

# ===================== Data Helpers =====================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file), None
        else:
            return pd.read_excel(file), None
    except Exception as e:
        return None, str(e)

@st.cache_data
def detect_outliers_counts(df, numeric_cols):
    out_counts = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            out_counts[col] = {"IQR": 0, "Z": 0, "ModZ": 0}
            continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1 if pd.notnull(Q3) and pd.notnull(Q1) else 0
        iqr_ct = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum()) if IQR != 0 else 0
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        z_ct = int((z > 3).sum()) if isinstance(z, np.ndarray) else 0
        med = np.median(s)
        mad = np.median(np.abs(s - med)) if len(s) else 0
        if mad == 0:
            modz_ct = 0
        else:
            modz = 0.6745 * (s - med) / mad
            modz_ct = int((np.abs(modz) > 3.5).sum())
        out_counts[col] = {"IQR": iqr_ct, "Z": z_ct, "ModZ": modz_ct}
    return out_counts

def safe_mode_fill(series):
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    return series.median() if pd.api.types.is_numeric_dtype(series) else "Unknown"

def build_data_brief(df: pd.DataFrame, max_cats=12) -> str:
    lines = []
    lines.append(f"ROWS={len(df)}, COLS={len(df.columns)}")
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    lines.append("DTYPES=" + "; ".join([f"{c}:{t}" for c, t in dtypes.items()]))
    miss = df.isnull().sum()
    miss_pct = (miss/len(df)*100).round(2)
    if len(miss[miss>0]) > 0:
        lines.append("MISSING=" + "; ".join([f"{c}:{int(miss[c])} ({miss_pct[c]}%)" for c in miss.index if miss[c]>0]))
    else:
        lines.append("MISSING=None")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]
    if num_cols:
        lines.append("NUMERIC_SUMMARY=")
        for c in num_cols:
            s = df[c].dropna()
            if s.empty: 
                continue
            lines.append(f"  {c}: min={s.min():.3g}, q1={s.quantile(.25):.3g}, med={s.median():.3g}, q3={s.quantile(.75):.3g}, max={s.max():.3g}, mean={s.mean():.3g}, std={s.std():.3g}, skew={s.skew():.3g}")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()[:8]
    if cat_cols:
        lines.append("CATEGORICAL_TOPS=")
        for c in cat_cols:
            vc = df[c].astype(str).value_counts(dropna=True).head(5)
            joined = ", ".join([f"{i}:{int(v)}" for i, v in vc.items()])
            lines.append(f"  {c}: {joined}")
    if num_cols:
        outs = detect_outliers_counts(df, num_cols)
        lines.append("OUTLIERS_SNAPSHOT=" + "; ".join([f"{c}(IQR={o['IQR']},Z={o['Z']},MZ={o['ModZ']})" for c, o in outs.items()]))
    return "\n".join(lines)

def ensure_gemini(model_name="gemini-1.5-flash"):
    key = st.session_state.gemini_key or os.getenv("GEMINI_API_KEY")
    if not key:
        st.warning("⚠️ Gemini API key is missing. Add it in the sidebar or set GEMINI_API_KEY env var.")
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        if st.session_state.gemini_chat is None:
            st.session_state.gemini_chat = model.start_chat(history=[
                {"role": "user", "parts": ["You are a data analyst assistant. Answer concisely with clear bullet points and, when helpful, short formulas. If unsure, ask a scoped follow-up."]},
                {"role": "model", "parts": ["Understood. I'll analyze the provided dataset context."]},
            ])
        return st.session_state.gemini_chat
    except Exception as e:
        st.error(f"Gemini init error: {e}")
        return None

# ===================== Main Auth Gate =====================
if st.session_state.auth_attempts >= st.session_state.max_auth_attempts and not st.session_state.authenticated:
    st.error("🚫 Maximum authentication attempts exceeded. Please refresh the page to try again.")
    st.stop()

if not st.session_state.authenticated:
    st.title("🔐 Secure Data Profiling App")
    st.markdown("### Please authenticate to access the application")

    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "👤 Register"])
    with auth_tab2:
        register_user_face()
    with auth_tab1:
        face_login()

    st.stop()

# ===================== Main App =====================
st.title(f"📊 Advanced Data Profiling & Cleaning App + 🤖 Gemini Chat")
st.markdown(f"*Welcome back, **{st.session_state.current_user}**!* 🎉")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Configuration")
show_auth_status()
st.sidebar.markdown("---")
st.session_state.gemini_key = st.sidebar.text_input(
    "Gemini API Key (optional if set via env)", type="password", value=st.session_state.gemini_key or ""
)
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="file_uploader")

if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
    for k in ["processed_data", "original_data", "last_uploaded_file", "data_brief"]:
        st.session_state[k] = None
    st.rerun()

# ===================== Load Data =====================
if uploaded_file:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.processed_data = None
        st.session_state.original_data = None
        st.session_state.last_uploaded_file = uploaded_file.name

    with st.spinner("Loading dataset..."):
        df, err = load_data(uploaded_file)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    if st.session_state.original_data is None:
        st.session_state.original_data = df.copy()
        st.session_state.processed_data = df.copy()
        st.session_state.data_brief = build_data_brief(df)

    st.success("✅ File loaded")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📋 Overview", "🔍 Data Quality", "🛠 Cleaning", "📊 Visualizations", "💾 Export", "🤖 AI Chat"]
    )

    # ---------- Overview ----------
    with tab1:
        cur = st.session_state.processed_data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(cur):,}")
        c2.metric("Columns", cur.shape[1])
        c3.metric("Memory (MB)", f"{cur.memory_usage(deep=True).sum()/1024**2:.2f}")
        c4.metric("Numeric Cols", len(cur.select_dtypes(include=np.number).columns))

        st.subheader("👀 Preview (editable)")
        show_n = st.slider("Rows to show", 5, min(1000, len(cur)), 10)
        edited = st.data_editor(cur.head(show_n), num_rows="dynamic", use_container_width=True, hide_index=True)
        if not edited.equals(cur.head(show_n)):
            st.session_state.processed_data.update(edited)
            st.session_state.data_brief = build_data_brief(st.session_state.processed_data)
            st.success("✅ Edits applied to in-memory data")

        st.subheader("📊 Column Info")
        info = pd.DataFrame({
            "dtype": cur.dtypes.astype(str),
            "non_null": cur.count(),
            "null": cur.isnull().sum(),
            "null_%": (cur.isnull().sum()/len(cur)*100).round(2),
            "unique": cur.nunique(),
            "unique_%": (cur.nunique()/len(cur)*100).round(2)
        })
        st.dataframe(info, use_container_width=True)

    # ---------- Data Quality ----------
    with tab2:
        cur = st.session_state.processed_data
        st.subheader("🕳️ Missing Values")
        miss = cur.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(x=miss.index, y=miss.values,
                             labels={"x": "Columns", "y": "Missing Count"},
                             title="Missing by Column")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(pd.DataFrame({"Missing": miss, "Missing_%": (miss/len(cur)*100).round(2)}))
        else:
            st.success("No missing values 🎉")

        st.subheader("🔄 Duplicates")
        dups = cur.duplicated().sum()
        if dups:
            st.warning(f"Found {dups} duplicate rows ({dups/len(cur)*100:.2f}%)")
            if st.button("Show Duplicates"):
                st.dataframe(cur[cur.duplicated(keep=False)].sort_values(cur.columns.tolist()), use_container_width=True)
        else:
            st.success("No duplicate rows ✅")

        st.subheader("🚨 Outlier Snapshot")
        num_cols = cur.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            out_counts = detect_outliers_counts(cur, num_cols)
            out_df = pd.DataFrame(out_counts).T
            st.dataframe(out_df, use_container_width=True)
            sel = st.selectbox("Box plot column", num_cols)
            if sel:
                st.plotly_chart(px.box(cur, y=sel, title=f"Box - {sel}"), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    # ---------- Cleaning ----------
    with tab3:
        wrk = st.session_state.processed_data.copy()

        st.markdown("### 🕳️ Handle Missing Values")
        if wrk.isnull().sum().sum() > 0:
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                miss_method = st.selectbox(
                    "Strategy",
                    ["None", "Drop rows", "Drop columns", "Fill mean", "Fill median", "Fill mode", "Forward fill", "Backward fill"]
                )
            with mcol2:
                thresh_pct = st.slider("Drop-column keep-threshold (%)", 0, 100, 50)

            if miss_method != "None" and st.button("Apply Missing Strategy"):
                if miss_method == "Drop rows":
                    wrk = wrk.dropna()
                elif miss_method == "Drop columns":
                    thresh = len(wrk) * (100 - thresh_pct)/100
                    wrk = wrk.dropna(axis=1, thresh=thresh)
                elif miss_method == "Fill mean":
                    nc = wrk.select_dtypes(include=np.number).columns
                    wrk[nc] = wrk[nc].fillna(wrk[nc].mean())
                elif miss_method == "Fill median":
                    nc = wrk.select_dtypes(include=np.number).columns
                    wrk[nc] = wrk[nc].fillna(wrk[nc].median())
                elif miss_method == "Fill mode":
                    for c in wrk.columns:
                        wrk[c] = wrk[c].fillna(safe_mode_fill(wrk[c]))
                elif miss_method == "Forward fill":
                    wrk = wrk.fillna(method="ffill")
                elif miss_method == "Backward fill":
                    wrk = wrk.fillna(method="bfill")

                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("Missing-value strategy applied ✅")
                st.rerun()
        else:
            st.success("No missing values to handle ✅")

        st.markdown("### 🚨 Outlier Handling")
        num_cols = wrk.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            cols = st.multiselect("Columns", num_cols)
            method = st.selectbox("Detection", ["IQR", "Z-Score", "Modified Z-Score"])
            treat = st.selectbox("Treatment", ["Remove", "Cap to bounds", "Replace with median"])

            if cols and st.button("Apply Outlier Treatment"):
                for c in cols:
                    s = wrk[c]
                    if method == "IQR":
                        Q1, Q3 = s.quantile(.25), s.quantile(.75); IQR = Q3 - Q1
                        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                        mask = (s < lb) | (s > ub)
                    elif method == "Z-Score":
                        z = np.abs(stats.zscore(s.dropna()))
                        mask = pd.Series(False, index=s.index)
                        mask.loc[s.dropna().index] = z > 3
                        lb, ub = None, None
                    else:
                        med = s.median(); mad = np.median(np.abs(s - med))
                        if mad == 0:
                            mask = pd.Series(False, index=s.index); lb, ub = None, None
                        else:
                            mz = 0.6745*(s - med)/mad
                            mask = np.abs(mz) > 3.5
                            lb, ub = None, None

                    if treat == "Remove":
                        wrk = wrk[~mask]
                    elif treat == "Cap to bounds" and method == "IQR":
                        wrk[c] = np.where(wrk[c] < lb, lb, np.where(wrk[c] > ub, ub, wrk[c]))
                    elif treat == "Replace with median":
                        wrk.loc[mask, c] = s.median()

                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("Outliers handled ✅")
                st.rerun()
        else:
            st.info("No numeric columns.")

        st.markdown("### 🔢 Encoding")
        cat_cols = wrk.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            enc_cols = st.multiselect("Categorical columns", cat_cols)
            enc_method = st.selectbox("Method", ["Label Encoding", "One-Hot Encoding"])
            if enc_cols and st.button("Apply Encoding"):
                if enc_method == "Label Encoding":
                    le = LabelEncoder()
                    for c in enc_cols:
                        wrk[c] = le.fit_transform(wrk[c].astype(str))
                else:
                    wrk = pd.get_dummies(wrk, columns=enc_cols, drop_first=True)

                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("Encoding applied ✅")
                st.rerun()
        else:
            st.info("No categorical columns.")

        st.markdown("### 📏 Scaling")
        num_cols = st.session_state.processed_data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sc_cols = st.multiselect("Columns to scale", num_cols)
            sc_method = st.selectbox("Scaler", ["StandardScaler (Z-score)", "MinMaxScaler (0-1)"])
            if sc_cols and st.button("Apply Scaling"):
                scaler = StandardScaler() if sc_method.startswith("Standard") else MinMaxScaler()
                wrk = st.session_state.processed_data.copy()
                wrk[sc_cols] = scaler.fit_transform(wrk[sc_cols])
                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("Scaling applied ✅")
                st.rerun()

        st.markdown("### 📋 Cleaning Summary")
        oc, pc = st.columns(2)
        oc.metric("Original Rows", f"{st.session_state.original_data.shape[0]:,}")
        oc.metric("Original Cols", st.session_state.original_data.shape[1])
        pc.metric("Current Rows", f"{st.session_state.processed_data.shape[0]:,}")
        pc.metric("Current Cols", st.session_state.processed_data.shape[1])

    # ---------- Visualizations ----------
    with tab4:
        cur = st.session_state.processed_data
        num_cols = cur.select_dtypes(include=np.number).columns.tolist()
        cat_cols = cur.select_dtypes(include=["object", "category"]).columns.tolist()
        kind = st.selectbox("Visualization Type", ["Distribution", "Correlation", "Categorical", "Custom"])

        if kind == "Distribution" and num_cols:
            cols = st.multiselect("Numeric columns", num_cols, default=num_cols[:min(4, len(num_cols))])
            ptype = st.radio("Plot", ["Histogram", "Box", "Violin"], horizontal=True)
            for c in cols:
                if ptype == "Histogram":
                    st.plotly_chart(px.histogram(cur, x=c, nbins=30, title=f"Distribution: {c}"), use_container_width=True)
                elif ptype == "Box":
                    st.plotly_chart(px.box(cur, y=c, title=f"Box: {c}"), use_container_width=True)
                else:
                    st.plotly_chart(px.violin(cur, y=c, title=f"Violin: {c}"), use_container_width=True)

        elif kind == "Correlation" and len(num_cols) > 1:
            corr = cur[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Correlation Heatmap"),
                            use_container_width=True)

        elif kind == "Categorical" and cat_cols:
            c = st.selectbox("Categorical column", cat_cols)
            vc = cur[c].astype(str).value_counts()
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.bar(x=vc.index, y=vc.values, labels={"x": c, "y": "count"}, title=f"Counts: {c}"),
                              use_container_width=True)
            col2.plotly_chart(px.pie(values=vc.values, names=vc.index, title=f"Share: {c}"),
                              use_container_width=True)

        elif kind == "Custom":
            x = st.selectbox("X", cur.columns)
            y = st.selectbox("Y", cur.columns, index=min(1, len(cur.columns)-1))
            color = st.selectbox("Color (optional)", [None] + list(cur.columns))
            p = st.selectbox("Plot", ["Scatter", "Line", "Bar"])
            if p == "Scatter":
                fig = px.scatter(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            elif p == "Line":
                fig = px.line(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            else:
                fig = px.bar(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Export ----------
    with tab5:
        cur = st.session_state.processed_data
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(cur):,}")
        c2.metric("Cols", cur.shape[1])
        c3.metric("Est. Size (MB)", f"{cur.memory_usage(deep=True).sum()/1024**2:.2f}")

        fmt = st.selectbox("Format", ["CSV", "Excel", "JSON", "Parquet", "HTML"])
        base = st.text_input("File name (no ext.)", f"cleaned_{(st.session_state.last_uploaded_file or 'data').split('.')[0]}")

        if fmt == "CSV":
            data, mime, ext = cur.to_csv(index=False), "text/csv", ".csv"
        elif fmt == "Excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                cur.to_excel(w, index=False, sheet_name="Cleaned")
            data, mime, ext = buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
        elif fmt == "JSON":
            data, mime, ext = cur.to_json(orient="records", indent=2), "application/json", ".json"
        elif fmt == "Parquet":
            buf = io.BytesIO()
            cur.to_parquet(buf, index=False)
            data, mime, ext = buf.getvalue(), "application/octet-stream", ".parquet"
        else:
            data, mime, ext = cur.to_html(index=False), "text/html", ".html"

        st.download_button(f"📥 Download {fmt}", data=data, file_name=f"{base}{ext}", mime=mime, use_container_width=True)

    # ---------- AI Chat ----------
    with tab6:
        st.subheader("🤖 Dataset-aware AI Chat (Gemini)")
        st.caption("Ask questions about your uploaded/cleaned dataset. The model uses a compact profile of your data, not the entire file.")
        chat = ensure_gemini()
        if chat is None:
            st.stop()

        with st.expander("📄 Data Brief sent to Gemini (read-only)", expanded=False):
            st.code(st.session_state.data_brief, language="text")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for role, msg in st.session_state.chat_messages:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Gemini:** {msg}")

        q = st.text_input("Ask about the data (e.g., outliers, missing, correlations, summaries):", key="gemini_q")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            send = st.button("Ask Gemini", type="primary", use_container_width=True)
        with col_b:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.gemini_chat = None
                st.session_state.chat_messages = []
                chat = ensure_gemini()
                st.rerun()

        if send and q:
            cur = st.session_state.processed_data
            brief = build_data_brief(cur)
            st.session_state.data_brief = brief
            system_prompt = f"""
You are a senior data analyst. You answer based on the user's question and the dataset brief provided below.
- If the user asks for column-specific calculations, reference column names exactly.
- If a direct numeric answer isn't possible from the brief, explain what additional step the user can run in the app (e.g., 'check Visualizations > Correlation' or 'apply outlier treatment IQR').
- Prefer concise bullet points and short formulas when useful.
- Never invent columns that don't exist.

DATASET BRIEF:
{brief}
"""
            try:
                resp = chat.send_message([system_prompt, f"USER QUESTION: {q}"])
                answer = resp.text
            except Exception as e:
                answer = f"Error from Gemini: {e}"

            st.session_state.chat_messages.append(("user", q))
            st.session_state.chat_messages.append(("assistant", answer))
            st.rerun()

# ===================== Empty State =====================
else:
    st.markdown("""
    ## Welcome! 🎉
    Upload a CSV/Excel from the sidebar to begin. Or try a sample:
    """)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load Sample Iris"):
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
            np.random.seed(7)
            df.loc[np.random.choice(df.index, 10), "sepal length (cm)"] = np.nan
            df.loc[np.random.choice(df.index, 5), "petal width (cm)"] = 10.0
            st.session_state.original_data = df.copy()
            st.session_state.processed_data = df.copy()
            st.session_state.last_uploaded_file = "iris_sample.csv"
            st.session_state.data_brief = build_data_brief(df)
            st.success("Sample loaded ✅")
            st.rerun()
    with c2:
        if st.button("Clear All Data"):
            for k in ["processed_data", "original_data", "last_uploaded_file", "data_brief"]:
                st.session_state[k] = None
            st.success("Cleared ✅")
            st.rerun()
