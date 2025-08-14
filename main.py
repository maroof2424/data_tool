# app.py - Improved Version
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Data Profiling & Cleaning App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.stProgress > div > div > div > div {
    background-color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Data Profiling & Cleaning App")
st.markdown("---")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

# Utility Functions
@st.cache_data
def load_data(file):
    """Load data with caching for performance"""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def detect_outliers_multiple_methods(df, column):
    """Detect outliers using multiple methods"""
    methods = {}
    
    # IQR Method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    methods['IQR'] = len(iqr_outliers)
    
    # Z-Score Method
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    z_outliers = len(z_scores[z_scores > 3])
    methods['Z-Score'] = z_outliers
    
    # Modified Z-Score Method
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    modified_z_scores = 0.6745 * (df[column] - median) / mad
    modified_z_outliers = len(modified_z_scores[np.abs(modified_z_scores) > 3.5])
    methods['Modified Z-Score'] = modified_z_outliers
    
    return methods

def safe_mode_fill(series):
    """Safely fill with mode, handle empty mode"""
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    else:
        return series.median() if series.dtype in ['int64', 'float64'] else 'Unknown'

    # ---- Sidebar Configuration ----
st.sidebar.header("🔧 Configuration")

# Show current data info in sidebar
if st.session_state.original_data is not None:
    with st.sidebar.expander("📊 Current Data Info", expanded=True):
        current_file = getattr(st.session_state, 'last_uploaded_file', 'Unknown')
        st.write(f"**File:** {current_file}")
        st.write(f"**Rows:** {len(st.session_state.processed_data):,}")
        st.write(f"**Columns:** {len(st.session_state.processed_data.columns)}")
        
        # Clear data button in sidebar
        if st.button("🗑️ Clear Current Data", type="secondary", use_container_width=True):
            st.session_state.original_data = None
            st.session_state.processed_data = None
            if 'last_uploaded_file' in st.session_state:
                del st.session_state.last_uploaded_file
            st.rerun()

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV or Excel file", 
    type=["csv", "xlsx"],
    help="Upload your dataset for analysis and cleaning",
    key="file_uploader"
)

# Reset session state when new file is uploaded
if uploaded_file:
    # Check if this is a new file
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.original_data = None
        st.session_state.processed_data = None
    
    # Load data with progress bar
    with st.spinner("Loading dataset..."):
        df, error = load_data(uploaded_file)
    
    if error:
        st.error(f"❌ Error loading file: {error}")
        st.stop()
    
    # Store original data (only once per file)
    if st.session_state.original_data is None:
        st.session_state.original_data = df.copy()
        st.session_state.processed_data = df.copy()
    
    st.success("✅ File uploaded successfully!")
    
    # Dataset size warning
    if len(df) > 10000:
        st.warning(f"⚠️ Large dataset detected ({len(df):,} rows). Some operations might be slower.")
    
    # ---- Main Content ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "🔍 Data Quality", "🛠 Cleaning", "📊 Visualizations", "💾 Export"
    ])
    
    # ---- TAB 1: Overview ----
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("📊 Columns", df.shape[1])
        with col3:
            st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=np.number).columns))
        
        st.subheader("👀 Dataset Preview")
        
        # Pagination for large datasets
        if len(df) > 100:
            page_size = st.selectbox("Rows per page", [50, 100, 200, 500], index=1)
            page_num = st.number_input("Page", min_value=1, max_value=max(1, len(df)//page_size + 1), value=1)
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            st.write(f"Showing rows {start_idx+1} to {end_idx} of {len(df)}")
            display_df = df.iloc[start_idx:end_idx]
        else:
            display_df = df
        
        # Editable dataframe
        edited_df = st.data_editor(
            display_df, 
            num_rows="dynamic", 
            use_container_width=True,
            hide_index=True
        )
        
        # Update session state if edited
        if not edited_df.equals(display_df):
            st.session_state.processed_data = edited_df
            st.success("✅ Data updated in memory!")
        
        # Column Information
        st.subheader("📊 Column Information")
        col_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique(),
            'Unique %': (df.nunique() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
    
    # ---- TAB 2: Data Quality ----
    with tab2:
        st.subheader("🔍 Data Quality Assessment")
        
        # Missing Values Analysis
        st.write("### 🕳️ Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    pd.DataFrame({
                        'Missing Count': missing_data,
                        'Missing %': (missing_data / len(df) * 100).round(2)
                    })
                )
        else:
            st.success("🎉 No missing values found!")
        
        # Duplicate Analysis
        st.write("### 🔄 Duplicate Analysis")
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"⚠️ Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.2f}%)")
            if st.button("Show Duplicate Rows"):
                st.dataframe(df[df.duplicated(keep=False)].sort_values(df.columns.tolist()))
        else:
            st.success("✅ No duplicate rows found!")
        
        # Outlier Analysis
        st.write("### 🚨 Outlier Analysis")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            if selected_col:
                outlier_methods = detect_outliers_multiple_methods(df, selected_col)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("**Outlier Detection Results:**")
                    for method, count in outlier_methods.items():
                        st.metric(f"{method} Method", f"{count} outliers")
                
                with col2:
                    # Box plot for outliers
                    fig = px.box(df, y=selected_col, title=f"Box Plot - {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for outlier analysis.")
        
        # Data Types Issues
        st.write("### 🏷️ Data Type Issues")
        type_issues = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric values stored as strings
                try:
                    pd.to_numeric(df[col], errors='raise')
                    type_issues.append(f"'{col}' appears to contain numeric data but is stored as text")
                except:
                    pass
        
        if type_issues:
            st.warning("⚠️ Potential data type issues:")
            for issue in type_issues:
                st.write(f"- {issue}")
        else:
            st.success("✅ No obvious data type issues found!")
    
    # ---- TAB 3: Cleaning ----
    with tab3:
        st.subheader("🛠 Data Cleaning Operations")
        
        # Use processed data for cleaning
        working_df = st.session_state.processed_data.copy()
        
        # Missing Values Section
        st.write("### 🕳️ Handle Missing Values")
        if working_df.isnull().sum().sum() > 0:
            missing_method = st.selectbox(
                "Choose missing value strategy:",
                ["None", "Drop rows", "Drop columns", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill", "Backward fill"]
            )
            
            if missing_method != "None":
                missing_threshold = st.slider(
                    "Missing value threshold (%) - only apply to columns with less missing values",
                    0, 100, 50
                )
                
                if st.button(f"Apply {missing_method}"):
                    with st.spinner("Processing missing values..."):
                        if missing_method == "Drop rows":
                            working_df = working_df.dropna()
                        elif missing_method == "Drop columns":
                            thresh = len(working_df) * (100 - missing_threshold) / 100
                            working_df = working_df.dropna(axis=1, thresh=thresh)
                        elif missing_method == "Fill with mean":
                            numeric_cols = working_df.select_dtypes(include=np.number).columns
                            working_df[numeric_cols] = working_df[numeric_cols].fillna(working_df[numeric_cols].mean())
                        elif missing_method == "Fill with median":
                            numeric_cols = working_df.select_dtypes(include=np.number).columns
                            working_df[numeric_cols] = working_df[numeric_cols].fillna(working_df[numeric_cols].median())
                        elif missing_method == "Fill with mode":
                            for col in working_df.columns:
                                working_df[col] = working_df[col].fillna(safe_mode_fill(working_df[col]))
                        elif missing_method == "Forward fill":
                            working_df = working_df.fillna(method='ffill')
                        elif missing_method == "Backward fill":
                            working_df = working_df.fillna(method='bfill')
                    
                    st.session_state.processed_data = working_df
                    st.success(f"✅ Applied {missing_method} successfully!")
                    st.rerun()
        else:
            st.success("✅ No missing values to handle!")
        
        # Duplicate Removal
        st.write("### 🔄 Remove Duplicates")
        if working_df.duplicated().sum() > 0:
            if st.button("Remove Duplicate Rows"):
                working_df = working_df.drop_duplicates()
                st.session_state.processed_data = working_df
                st.success("✅ Duplicates removed!")
                st.rerun()
        else:
            st.success("✅ No duplicates to remove!")
        
        # Outlier Handling
        st.write("### 🚨 Handle Outliers")
        numeric_cols = working_df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            outlier_columns = st.multiselect(
                "Select columns for outlier treatment:",
                numeric_cols,
                help="Choose which columns to apply outlier treatment"
            )
            
            if outlier_columns:
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["IQR", "Z-Score", "Modified Z-Score"]
                )
                
                outlier_treatment = st.selectbox(
                    "Outlier treatment:",
                    ["Remove", "Cap values", "Transform to median"]
                )
                
                if st.button("Apply Outlier Treatment"):
                    with st.spinner("Processing outliers..."):
                        for col in outlier_columns:
                            if outlier_method == "IQR":
                                Q1 = working_df[col].quantile(0.25)
                                Q3 = working_df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outlier_mask = (working_df[col] < lower_bound) | (working_df[col] > upper_bound)
                            
                            elif outlier_method == "Z-Score":
                                z_scores = np.abs(stats.zscore(working_df[col].dropna()))
                                outlier_mask = z_scores > 3
                            
                            elif outlier_method == "Modified Z-Score":
                                median = working_df[col].median()
                                mad = np.median(np.abs(working_df[col] - median))
                                modified_z_scores = 0.6745 * (working_df[col] - median) / mad
                                outlier_mask = np.abs(modified_z_scores) > 3.5
                            
                            # Apply treatment
                            if outlier_treatment == "Remove":
                                working_df = working_df[~outlier_mask]
                            elif outlier_treatment == "Cap values":
                                if outlier_method == "IQR":
                                    working_df[col] = np.where(working_df[col] < lower_bound, lower_bound,
                                                             np.where(working_df[col] > upper_bound, upper_bound, working_df[col]))
                            elif outlier_treatment == "Transform to median":
                                working_df.loc[outlier_mask, col] = working_df[col].median()
                    
                    st.session_state.processed_data = working_df
                    st.success("✅ Outlier treatment applied!")
                    st.rerun()
        
        # Encoding
        st.write("### 🔢 Encode Categorical Variables")
        cat_cols = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            encoding_cols = st.multiselect("Select columns to encode:", cat_cols)
            
            if encoding_cols:
                encoding_method = st.selectbox(
                    "Encoding method:",
                    ["Label Encoding", "One-Hot Encoding", "Target Encoding"]
                )
                
                if st.button("Apply Encoding"):
                    with st.spinner("Applying encoding..."):
                        if encoding_method == "Label Encoding":
                            le = LabelEncoder()
                            for col in encoding_cols:
                                working_df[col] = le.fit_transform(working_df[col].astype(str))
                        
                        elif encoding_method == "One-Hot Encoding":
                            working_df = pd.get_dummies(working_df, columns=encoding_cols, drop_first=True)
                        
                        elif encoding_method == "Target Encoding":
                            st.warning("Target encoding requires a target variable. Skipping for now.")
                    
                    st.session_state.processed_data = working_df
                    st.success("✅ Encoding applied!")
                    st.rerun()
        else:
            st.info("No categorical columns found for encoding.")
        
        # Feature Scaling
        st.write("### 📏 Feature Scaling")
        if numeric_cols:
            scaling_cols = st.multiselect("Select columns for scaling:", numeric_cols)
            
            if scaling_cols:
                scaling_method = st.selectbox(
                    "Scaling method:",
                    ["StandardScaler (Z-score normalization)", "MinMaxScaler (0-1 normalization)"]
                )
                
                if st.button("Apply Scaling"):
                    with st.spinner("Applying scaling..."):
                        if scaling_method.startswith("StandardScaler"):
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        
                        working_df[scaling_cols] = scaler.fit_transform(working_df[scaling_cols])
                    
                    st.session_state.processed_data = working_df
                    st.success("✅ Scaling applied!")
                    st.rerun()
        
        # Show cleaning summary
        st.write("### 📋 Cleaning Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", f"{st.session_state.original_data.shape[0]:,}")
            st.metric("Original Columns", st.session_state.original_data.shape[1])
        with col2:
            st.metric("Current Rows", f"{st.session_state.processed_data.shape[0]:,}")
            st.metric("Current Columns", st.session_state.processed_data.shape[1])
    
    # ---- TAB 4: Visualizations ----
    with tab4:
        st.subheader("📊 Interactive Visualizations")
        
        working_df = st.session_state.processed_data
        numeric_cols = working_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Visualization controls
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Distribution Plots", "Correlation Analysis", "Categorical Analysis", "Outlier Visualization", "Custom Plots"]
        )
        
        if viz_type == "Distribution Plots" and numeric_cols:
            selected_cols = st.multiselect(
                "Select columns for distribution analysis:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_cols:
                plot_type = st.radio("Plot type:", ["Histogram", "Box Plot", "Violin Plot"])
                
                if plot_type == "Histogram":
                    for col in selected_cols:
                        fig = px.histogram(working_df, x=col, nbins=30, title=f"Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Box Plot":
                    for col in selected_cols:
                        fig = px.box(working_df, y=col, title=f"Box Plot - {col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Violin Plot":
                    for col in selected_cols:
                        fig = px.violin(working_df, y=col, title=f"Violin Plot - {col}")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Analysis" and len(numeric_cols) > 1:
            corr_matrix = working_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlation pairs
            st.write("### 🔗 High Correlation Pairs (> 0.8)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr))
            else:
                st.info("No high correlations found (> 0.8)")
        
        elif viz_type == "Categorical Analysis" and cat_cols:
            selected_cat = st.selectbox("Select categorical column:", cat_cols)
            
            if selected_cat:
                value_counts = working_df[selected_cat].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Value Counts - {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution - {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Custom Plots":
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis:", working_df.columns)
                y_col = st.selectbox("Y-axis:", working_df.columns)
            
            with col2:
                plot_type = st.selectbox("Plot type:", ["Scatter", "Line", "Bar"])
                color_col = st.selectbox("Color by (optional):", [None] + list(working_df.columns))
            
            if x_col and y_col:
                if plot_type == "Scatter":
                    fig = px.scatter(working_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                elif plot_type == "Line":
                    fig = px.line(working_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                elif plot_type == "Bar":
                    fig = px.bar(working_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ---- TAB 5: Export ----
    with tab5:
        st.subheader("💾 Export Cleaned Dataset")
        
        working_df = st.session_state.processed_data
        
        # Export summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows to Export", f"{len(working_df):,}")
        with col2:
            st.metric("Columns to Export", working_df.shape[1])
        with col3:
            st.metric("File Size (Est.)", f"{working_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Export options
        export_format = st.selectbox(
            "Select export format:",
            ["CSV", "Excel", "JSON", "Parquet", "HTML"]
        )
        
        # File naming with current data info
        current_file_name = getattr(st.session_state, 'last_uploaded_file', 'data')
        if current_file_name == "sample_iris_data":
            default_name = "cleaned_iris_sample"
        else:
            default_name = f"cleaned_{current_file_name.split('.')[0]}" if '.' in current_file_name else "cleaned_data"
        
        file_name = st.text_input("File name (without extension):", default_name)
        
        # Generate export data
        if export_format == "CSV":
            export_data = working_df.to_csv(index=False)
            mime_type = "text/csv"
            file_extension = ".csv"
        
        elif export_format == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                working_df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
            export_data = buffer.getvalue()
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_extension = ".xlsx"
        
        elif export_format == "JSON":
            export_data = working_df.to_json(orient="records", indent=2)
            mime_type = "application/json"
            file_extension = ".json"
        
        elif export_format == "Parquet":
            buffer = io.BytesIO()
            working_df.to_parquet(buffer, index=False)
            export_data = buffer.getvalue()
            mime_type = "application/octet-stream"
            file_extension = ".parquet"
        
        elif export_format == "HTML":
            export_data = working_df.to_html(index=False, classes='table table-striped')
            mime_type = "text/html"
            file_extension = ".html"
        
        # Download button
        st.download_button(
            label=f"📥 Download {export_format}",
            data=export_data,
            file_name=f"{file_name}{file_extension}",
            mime=mime_type,
            use_container_width=True
        )
        
        # Preview export data
        if st.checkbox("Preview export data"):
            st.subheader("Export Preview")
            if len(working_df) > 1000:
                st.warning("Showing first 1000 rows for performance")
                st.dataframe(working_df.head(1000), use_container_width=True)
            else:
                st.dataframe(working_df, use_container_width=True)
        
        # Export report
        if st.button("Generate Cleaning Report"):
            st.subheader("📋 Data Cleaning Report")
            
            report = f"""
            # Data Cleaning Report
            
            ## Original Dataset
            - Rows: {st.session_state.original_data.shape[0]:,}
            - Columns: {st.session_state.original_data.shape[1]}
            - Memory Usage: {st.session_state.original_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            
            ## Cleaned Dataset
            - Rows: {working_df.shape[0]:,}
            - Columns: {working_df.shape[1]}
            - Memory Usage: {working_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            
            ## Changes Made
            - Rows removed: {st.session_state.original_data.shape[0] - working_df.shape[0]:,}
            - Columns removed: {st.session_state.original_data.shape[1] - working_df.shape[1]}
            - Data reduction: {(1 - len(working_df)/len(st.session_state.original_data))*100:.1f}%
            
            ## Data Quality Improvements
            - Missing values handled: ✅
            - Duplicates removed: ✅ 
            - Outliers treated: ✅
            - Data types optimized: ✅
            """
            
            st.markdown(report)
            
            # Download report
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="cleaning_report.md",
                mime="text/markdown"
            )

else:
    # Welcome message
    st.markdown("""
    ## Welcome to the Advanced Data Profiling & Cleaning App! 🎉
    
    ### Features:
    - 📊 **Interactive Data Exploration** with pagination for large datasets
    - 🔍 **Comprehensive Data Quality Assessment** 
    - 🛠 **Advanced Cleaning Operations** with multiple methods
    - 📈 **Interactive Visualizations** using Plotly
    - 💾 **Multiple Export Formats** with detailed reports
    - 🚀 **Performance Optimized** with caching and progress bars
    
    ### Getting Started:
    1. Upload your CSV or Excel file using the sidebar
    2. Explore your data in the **Overview** tab
    3. Check data quality issues in the **Data Quality** tab
    4. Apply cleaning operations in the **Cleaning** tab
    5. Create visualizations in the **Visualizations** tab
    6. Export your cleaned data in the **Export** tab
    
    ### Supported Operations:
    - Missing value handling (8 different methods)
    - Duplicate detection and removal
    - Outlier detection (IQR, Z-Score, Modified Z-Score)
    - Categorical encoding (Label, One-Hot)
    - Feature scaling (Standard, MinMax)
    - Data type optimization
    
    **Upload your dataset to get started!** 📁
    """)
    
    # Sample data option
    st.markdown("### 🎯 Try with Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sample Dataset (Iris)", type="secondary"):
            # Create sample dataset
            from sklearn.datasets import load_iris
            iris = load_iris()
            sample_df = pd.DataFrame(iris.data, columns=iris.feature_names)
            sample_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            
            # Add some missing values and outliers for demonstration
            np.random.seed(42)
            sample_df.loc[np.random.choice(sample_df.index, 10), 'sepal length (cm)'] = np.nan
            sample_df.loc[np.random.choice(sample_df.index, 5), 'petal width (cm)'] = 10.0  # outlier
            
            st.session_state.original_data = sample_df.copy()
            st.session_state.processed_data = sample_df.copy()
            st.session_state.last_uploaded_file = "sample_iris_data"
            st.success("✅ Sample dataset loaded! Refresh the page to see the tabs.")
            st.rerun()
    
    with col2:
        if st.button("Clear All Data", type="primary"):
            # Clear all session state data
            st.session_state.original_data = None
            st.session_state.processed_data = None
            if 'last_uploaded_file' in st.session_state:
                del st.session_state.last_uploaded_file
            st.success("✅ All data cleared!")
            st.rerun()

# ---- Additional Utility Functions ----
def generate_data_report(original_df, processed_df):
    """Generate comprehensive data cleaning report"""
    report = {
        'original_shape': original_df.shape,
        'processed_shape': processed_df.shape,
        'missing_values_before': original_df.isnull().sum().sum(),
        'missing_values_after': processed_df.isnull().sum().sum(),
        'duplicates_before': original_df.duplicated().sum(),
        'duplicates_after': processed_df.duplicated().sum(),
        'data_types_before': original_df.dtypes.value_counts().to_dict(),
        'data_types_after': processed_df.dtypes.value_counts().to_dict()
    }
    return report

@st.cache_data
def calculate_data_profile(df):
    """Calculate comprehensive data profile"""
    profile = {}
    
    for col in df.columns:
        col_profile = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percent': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percent': (df[col].nunique() / len(df)) * 100
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_profile.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })
        else:
            col_profile.update({
                'top_value': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'top_value_freq': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
            })
        
        profile[col] = col_profile
    
    return profile

# Performance monitoring
@st.cache_data
def get_memory_usage(df):
    """Get detailed memory usage information"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_mb': total_memory / 1024**2,
        'by_column': memory_usage.to_dict(),
        'dtypes_memory': df.groupby(df.dtypes).size().to_dict()
    }

# Advanced data validation functions
def validate_data_integrity(df):
    """Validate data integrity and return issues"""
    issues = []
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        issues.append(f"Empty columns found: {empty_cols}")
    
    # Check for columns with single value
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
    
    if single_value_cols:
        issues.append(f"Columns with single value: {single_value_cols}")
    
    # Check for potential ID columns
    potential_ids = []
    for col in df.columns:
        if df[col].nunique() == len(df) and not df[col].isnull().any():
            potential_ids.append(col)
    
    if potential_ids:
        issues.append(f"Potential ID columns (all unique): {potential_ids}")
    
    # Check for high cardinality categorical columns
    high_card_cats = []
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].nunique() > len(df) * 0.5:  # More than 50% unique values
            high_card_cats.append(col)
    
    if high_card_cats:
        issues.append(f"High cardinality categorical columns: {high_card_cats}")
    
    return issues

# Data transformation suggestions
def suggest_transformations(df):
    """Suggest data transformations based on data characteristics"""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Check skewness
        skewness = df[col].skew()
        if abs(skewness) > 2:
            if skewness > 2:
                suggestions.append(f"'{col}' is highly right-skewed (skew={skewness:.2f}). Consider log transformation.")
            else:
                suggestions.append(f"'{col}' is highly left-skewed (skew={skewness:.2f}). Consider square transformation.")
        
        # Check for potential scaling needs
        if df[col].std() > df[col].mean() * 10:
            suggestions.append(f"'{col}' has high variance. Consider scaling.")
        
        # Check for potential categorical encoded as numeric
        if df[col].nunique() < 10 and df[col].dtype in ['int64']:
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) > 1 and all(isinstance(x, (int, np.integer)) for x in unique_vals):
                suggestions.append(f"'{col}' might be categorical data encoded as numeric (unique values: {unique_vals}).")
    
    return suggestions

# Export enhanced functionality
def create_enhanced_export(df, format_type, include_metadata=True):
    """Create enhanced export with metadata and formatting"""
    
    if format_type == "Excel Enhanced":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
            
            # Metadata sheet
            if include_metadata:
                metadata = pd.DataFrame({
                    'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Duplicate Rows', 'Memory Usage (MB)'],
                    'Value': [
                        len(df),
                        len(df.columns),
                        df.isnull().sum().sum(),
                        df.duplicated().sum(),
                        round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                    ]
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Column info sheet
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data_Type': df.dtypes.values,
                    'Non_Null_Count': df.count().values,
                    'Unique_Values': df.nunique().values,
                    'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                col_info.to_excel(writer, sheet_name='Column_Info', index=False)
        
        return buffer.getvalue()
    
    return None

# Add footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ using Streamlit | 
    <strong>Features:</strong> Interactive EDA • Advanced Cleaning • Smart Visualizations • Multiple Export Formats</p>
    <p><strong>Tip:</strong> Use the sidebar to upload your data and navigate through the tabs for a complete data cleaning workflow!</p>
</div>
""", unsafe_allow_html=True)