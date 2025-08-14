# app.py — Advanced Data Profiling + Cleaning + Visualizations + Gemini Chatbot
import os, io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# Viz
import plotly.express as px

# ML utils
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Gemini
import google.generativeai as genai

# ------------------------- Page Config & Styles -------------------------
st.set_page_config(page_title="Advanced Data Profiling + Gemini Chat", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Advanced Data Profiling & Cleaning App + 🤖 Gemini Chat")
st.markdown("""
<style>
.metric-container { background:#f6f7fb; padding:10px; border-radius:12px; }
.small { color:#6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- Session State -------------------------
for k, v in {
    "processed_data": None,
    "original_data": None,
    "last_uploaded_file": None,
    "gemini_chat": None,
    "gemini_key": None,
    "data_brief": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------- Helpers -------------------------
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
        # IQR
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1 if pd.notnull(Q3) and pd.notnull(Q1) else 0
        iqr_ct = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum()) if IQR != 0 else 0
        # Z
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        z_ct = int((z > 3).sum()) if isinstance(z, np.ndarray) else 0
        # Modified Z
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
    """Compact profile string for LLM context."""
    lines = []
    lines.append(f"ROWS={len(df)}, COLS={len(df.columns)}")
    # dtypes
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    lines.append("DTYPES=" + "; ".join([f"{c}:{t}" for c, t in dtypes.items()]))

    # missing
    miss = df.isnull().sum()
    miss_pct = (miss/len(df)*100).round(2)
    if len(miss[miss>0]) > 0:
        lines.append("MISSING=" + "; ".join([f"{c}:{int(miss[c])} ({miss_pct[c]}%)" for c in miss.index if miss[c]>0]))
    else:
        lines.append("MISSING=None")

    # numeric stats (subset to 8 cols for brevity)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]
    if num_cols:
        lines.append("NUMERIC_SUMMARY=")
        for c in num_cols:
            s = df[c].dropna()
            if s.empty: continue
            lines.append(f"  {c}: min={s.min():.3g}, q1={s.quantile(.25):.3g}, med={s.median():.3g}, q3={s.quantile(.75):.3g}, max={s.max():.3g}, mean={s.mean():.3g}, std={s.std():.3g}, skew={s.skew():.3g}")

    # categorical top values
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()[:8]
    if cat_cols:
        lines.append("CATEGORICAL_TOPS=")
        for c in cat_cols:
            vc = df[c].astype(str).value_counts(dropna=True).head(5)
            joined = ", ".join([f"{i}:{int(v)}" for i, v in vc.items()])
            lines.append(f"  {c}: {joined}")

    # outlier counts snapshot
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
        # initialize chat session once
        if st.session_state.gemini_chat is None:
            st.session_state.gemini_chat = model.start_chat(history=[
                {"role": "user", "parts": ["You are a data analyst assistant. Answer concisely with clear bullet points and, when helpful, short formulas. If unsure, ask a scoped follow-up."]},
                {"role": "model", "parts": ["Understood. I'll analyze the provided dataset context."]},
            ])
        return st.session_state.gemini_chat
    except Exception as e:
        st.error(f"Gemini init error: {e}")
        return None

# ------------------------- Sidebar -------------------------
st.sidebar.header("🔧 Configuration")

# Gemini key input (optional if env var set)
st.session_state.gemini_key = st.sidebar.text_input("Gemini API Key (optional if set via env)", type="password", value=st.session_state.gemini_key or "")

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xlsx"], key="file_uploader")

# Clear btn
if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
    for k in ["processed_data","original_data","last_uploaded_file","data_brief"]:
        st.session_state[k] = None
    st.rerun()

# ------------------------- Load Data -------------------------
if uploaded_file:
    # reset if new file
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

    # --------------------- Tabs ---------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview", "🔍 Data Quality", "🛠 Cleaning", "📊 Visualizations", "💾 Export", "🤖 AI Chat"
    ])

    # ---------- TAB 1: Overview ----------
    with tab1:
        cur = st.session_state.processed_data
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{len(cur):,}")
        c2.metric("Columns", cur.shape[1])
        c3.metric("Memory (MB)", f"{cur.memory_usage(deep=True).sum()/1024**2:.2f}")
        c4.metric("Numeric Cols", len(cur.select_dtypes(include=np.number).columns))

        st.subheader("👀 Preview (editable)")
        show_n = st.slider("Rows to show", 5, min(1000, len(cur)), 10)
        edited = st.data_editor(cur.head(show_n), num_rows="dynamic", use_container_width=True, hide_index=True)
        if not edited.equals(cur.head(show_n)):
            # apply edits back to processed_data (head-only edits patched)
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

    # ---------- TAB 2: Data Quality ----------
    with tab2:
        cur = st.session_state.processed_data
        st.subheader("🕳️ Missing Values")
        miss = cur.isnull().sum()
        miss = miss[miss>0].sort_values(ascending=False)
        if len(miss):
            col1,col2 = st.columns([2,1])
            with col1:
                fig = px.bar(x=miss.index, y=miss.values, labels={"x":"Columns","y":"Missing Count"}, title="Missing by Column")
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

    # ---------- TAB 3: Cleaning ----------
    with tab3:
        wrk = st.session_state.processed_data.copy()

        st.markdown("### 🕳️ Handle Missing Values")
        if wrk.isnull().sum().sum() > 0:
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                miss_method = st.selectbox("Strategy", ["None","Drop rows","Drop columns","Fill mean","Fill median","Fill mode","Forward fill","Backward fill"])
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
            method = st.selectbox("Detection", ["IQR","Z-Score","Modified Z-Score"])
            treat = st.selectbox("Treatment", ["Remove","Cap to bounds","Replace with median"])
            if cols and st.button("Apply Outlier Treatment"):
                for c in cols:
                    s = wrk[c]
                    if method == "IQR":
                        Q1, Q3 = s.quantile(.25), s.quantile(.75); IQR = Q3-Q1
                        lb, ub = Q1-1.5*IQR, Q3+1.5*IQR
                        mask = (s<lb)|(s>ub)
                    elif method == "Z-Score":
                        z = np.abs(stats.zscore(s.dropna()))
                        mask = pd.Series(False, index=s.index)
                        mask.loc[s.dropna().index] = z > 3
                        lb, ub = None, None
                    else:
                        med = s.median(); mad = np.median(np.abs(s-med))
                        if mad == 0:
                            mask = pd.Series(False, index=s.index); lb, ub = None, None
                        else:
                            mz = 0.6745*(s-med)/mad
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
        cat_cols = wrk.select_dtypes(include=["object","category"]).columns.tolist()
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
            sc_method = st.selectbox("Scaler", ["StandardScaler (Z-score)","MinMaxScaler (0-1)"])
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

    # ---------- TAB 4: Visualizations ----------
    with tab4:
        cur = st.session_state.processed_data
        num_cols = cur.select_dtypes(include=np.number).columns.tolist()
        cat_cols = cur.select_dtypes(include=["object","category"]).columns.tolist()

        kind = st.selectbox("Visualization Type", ["Distribution","Correlation","Categorical","Custom"])

        if kind == "Distribution" and num_cols:
            cols = st.multiselect("Numeric columns", num_cols, default=num_cols[:min(4, len(num_cols))])
            ptype = st.radio("Plot", ["Histogram","Box","Violin"], horizontal=True)
            for c in cols:
                if ptype == "Histogram":
                    st.plotly_chart(px.histogram(cur, x=c, nbins=30, title=f"Distribution: {c}"), use_container_width=True)
                elif ptype == "Box":
                    st.plotly_chart(px.box(cur, y=c, title=f"Box: {c}"), use_container_width=True)
                else:
                    st.plotly_chart(px.violin(cur, y=c, title=f"Violin: {c}"), use_container_width=True)

        elif kind == "Correlation" and len(num_cols) > 1:
            corr = cur[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Correlation Heatmap"), use_container_width=True)

        elif kind == "Categorical" and cat_cols:
            c = st.selectbox("Categorical column", cat_cols)
            vc = cur[c].astype(str).value_counts()
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.bar(x=vc.index, y=vc.values, labels={"x":c,"y":"count"}, title=f"Counts: {c}"), use_container_width=True)
            col2.plotly_chart(px.pie(values=vc.values, names=vc.index, title=f"Share: {c}"), use_container_width=True)

        elif kind == "Custom":
            x = st.selectbox("X", cur.columns)
            y = st.selectbox("Y", cur.columns, index=min(1, len(cur.columns)-1))
            color = st.selectbox("Color (optional)", [None] + list(cur.columns))
            p = st.selectbox("Plot", ["Scatter","Line","Bar"])
            if p == "Scatter":
                fig = px.scatter(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            elif p == "Line":
                fig = px.line(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            else:
                fig = px.bar(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 5: Export ----------
    with tab5:
        cur = st.session_state.processed_data
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", f"{len(cur):,}")
        c2.metric("Cols", cur.shape[1])
        c3.metric("Est. Size (MB)", f"{cur.memory_usage(deep=True).sum()/1024**2:.2f}")

        fmt = st.selectbox("Format", ["CSV","Excel","JSON","Parquet","HTML"])
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
            buf = io.BytesIO(); cur.to_parquet(buf, index=False); data, mime, ext = buf.getvalue(), "application/octet-stream", ".parquet"
        else:
            data, mime, ext = cur.to_html(index=False), "text/html", ".html"

        st.download_button(f"📥 Download {fmt}", data=data, file_name=f"{base}{ext}", mime=mime, use_container_width=True)

    # ---------- TAB 6: 🤖 Gemini Chat ----------
    with tab6:
        st.subheader("🤖 Dataset-aware AI Chat (Gemini)")
        st.caption("Ask questions about your uploaded/cleaned dataset. The model uses a compact profile of your data, not the entire file.")

        # Ensure chat session
        chat = ensure_gemini()
        if chat is None:
            st.stop()

        # Show current brief & controls
        with st.expander("📄 Data Brief sent to Gemini (read-only)", expanded=False):
            st.code(st.session_state.data_brief, language="text")

        # Chat UI
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display history
        for role, msg in st.session_state.chat_messages:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Gemini:** {msg}")

        # Input
        q = st.text_input("Ask about the data (e.g., outliers, missing, correlations, summaries):", key="gemini_q")
        col_a, col_b = st.columns([1,1])
        with col_a:
            send = st.button("Ask Gemini", type="primary", use_container_width=True)
        with col_b:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.gemini_chat = None
                st.session_state.chat_messages = []
                chat = ensure_gemini()
                st.experimental_rerun()

        if send and q:
            cur = st.session_state.processed_data
            # Refresh brief to reflect latest cleaning
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
            st.experimental_rerun()

else:
    # Welcome / Sample
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
            # Add some NA and outliers
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
            for k in ["processed_data","original_data","last_uploaded_file","data_brief"]:
                st.session_state[k] = None
            st.success("Cleared ✅")
            st.rerun()
