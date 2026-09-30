import io
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(
    page_title="Survey Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .main, .stApp {
        background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 32%),
                    radial-gradient(circle at top right, rgba(124, 58, 237, 0.14), transparent 30%),
                    linear-gradient(180deg, #050712 0%, #0b1120 42%, #050712 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        letter-spacing: -0.03em;
    }

    p, label, span, div {
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #d1d5db !important;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        border-radius: 30px;
        padding: 42px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.30), transparent 34%),
            radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.18), transparent 32%),
            linear-gradient(135deg, #0f172a 0%, #1e1b4b 58%, #07111f 100%);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
        margin-bottom: 24px;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 8px 13px;
        font-size: 0.78rem;
        font-weight: 800;
        color: #bfdbfe;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.10);
    }

    .hero-title {
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: clamp(2.4rem, 5vw, 4.9rem);
        line-height: 0.95;
        font-weight: 950;
        color: #ffffff;
    }

    .hero-gradient {
        background: linear-gradient(90deg, #bfdbfe, #ddd6fe, #a7f3d0);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-subtitle {
        max-width: 850px;
        margin-top: 18px;
        color: #cbd5e1;
        font-size: 1.08rem;
        line-height: 1.8;
    }

    .upload-shell {
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(15, 23, 42, 0.72);
        border-radius: 30px;
        padding: 30px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
        margin-bottom: 24px;
    }

    .upload-title {
        text-align: center;
        font-size: 1.7rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .upload-subtitle {
        text-align: center;
        color: #94a3b8;
        max-width: 720px;
        margin: 0 auto 24px auto;
        line-height: 1.75;
    }

    .upload-features {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 18px;
    }

    .upload-feature {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.045);
        color: #cbd5e1;
        font-weight: 750;
        font-size: 0.92rem;
        text-align: center;
    }

    [data-testid="stFileUploader"] {
        max-width: 760px;
        margin: 0 auto;
    }

    [data-testid="stFileUploader"] section {
        min-height: 190px;
        border-radius: 28px !important;
        border: 2px dashed rgba(96, 165, 250, 0.65) !important;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(124, 58, 237, 0.10)) !important;
        transition: all 160ms ease;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(167, 243, 208, 0.88) !important;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(16, 185, 129, 0.12)) !important;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] button {
        border-radius: 999px !important;
        font-weight: 900 !important;
        padding: 0.65rem 1.35rem !important;
        border: 0 !important;
        background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
        color: white !important;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(15, 23, 42, 0.98));
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 14px 35px rgba(0,0,0,0.24);
    }

    .metric-title {
        font-size: 0.84rem;
        color: #94a3b8;
        margin-bottom: 8px;
        font-weight: 800;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        color: #f8fafc;
    }

    .metric-sub {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 6px;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .dataset-pill {
        border: 1px solid rgba(16, 185, 129, 0.30);
        background: rgba(16, 185, 129, 0.12);
        color: #bbf7d0;
        padding: 12px 14px;
        border-radius: 18px;
        margin-bottom: 8px;
        font-weight: 850;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(31, 41, 55, 0.75);
        border-radius: 14px 14px 0 0;
        padding: 12px 18px;
        color: #e5e7eb;
        border: 1px solid rgba(148, 163, 184, 0.16);
        font-weight: 800;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
    }

    .soft-note {
        padding: 18px 20px;
        border-radius: 22px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.24);
        color: #dbeafe;
        line-height: 1.75;
    }

    @media (max-width: 800px) {
        .hero-card, .upload-shell { padding: 24px; }
        .upload-features { grid-template-columns: 1fr; }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STOPWORDS = set("""
the and to of in a an is are was were for with on at by from this that these those it as be or if yes no not i you we they he she them their our your about into can could should would there here very more most less also have has had do does did because than then so such using use used assessment speaking course feedback formative summative learning teaching survey response responses question questions dataset data file files
""".split())


def clean_dataset_name(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() or "Dataset"


def make_unique_name(base_name: str, existing: dict) -> str:
    if base_name not in existing:
        return base_name
    counter = 2
    while f"{base_name} ({counter})" in existing:
        counter += 1
    return f"{base_name} ({counter})"


def make_unique_columns(columns):
    cleaned = (
        pd.Series(columns)
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
        .replace("", "Unnamed")
        .tolist()
    )
    seen = {}
    result = []
    for col in cleaned:
        seen[col] = seen.get(col, 0) + 1
        result.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    return result


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = file_name.lower()
    buffer = io.BytesIO(file_bytes)
    if lower_name.endswith(".csv"):
        try:
            df = pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin-1")
    elif lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type")
    return clean_dataframe(df)


def build_datasets(uploaded_files):
    datasets = {}
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.getvalue()
            df = load_uploaded_file(uploaded_file.name, file_bytes)
            base_name = clean_dataset_name(uploaded_file.name)
            dataset_name = make_unique_name(base_name, datasets)
            datasets[dataset_name] = {
                "filename": uploaded_file.name,
                "df": df,
                "size_kb": round(len(file_bytes) / 1024, 1),
            }
        except Exception as error:
            st.error(f"Could not load {uploaded_file.name}: {error}")
    return datasets


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def numeric_columns(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        converted = coerce_numeric_series(df[col])
        if converted.notna().sum() >= 2:
            cols.append(col)
    return cols


def categorical_columns(df: pd.DataFrame, max_unique=40):
    cols = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        if 1 < unique_count <= max_unique:
            cols.append(col)
    return cols


def likely_text_columns(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        avg_len = s.str.len().mean()
        unique_ratio = s.nunique() / max(len(s), 1)
        if avg_len > 25 or unique_ratio > 0.65:
            cols.append(col)
    return cols


def summary_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df[column].fillna("Missing").astype(str).str.strip()
    counts = data.value_counts(dropna=False)
    total = max(len(df), 1)
    return pd.DataFrame({
        "Response": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / total * 100).round(1),
    })


def cronbach_alpha(data: pd.DataFrame):
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    k = data.shape[1]
    if k < 2 or len(data) < 2:
        return None, len(data), k
    item_var = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0 or pd.isna(total_var):
        return None, len(data), k
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)
    return float(alpha), len(data), k


def alpha_label(alpha):
    if alpha is None:
        return "Not enough valid data"
    if alpha >= 0.90:
        return "Excellent reliability"
    if alpha >= 0.80:
        return "Good reliability"
    if alpha >= 0.70:
        return "Acceptable reliability"
    if alpha >= 0.60:
        return "Questionable reliability"
    return "Poor reliability"


def text_keywords(series: pd.Series, top_n=20):
    text = " ".join(series.dropna().astype(str).tolist()).lower()
    words = re.findall(r"[a-zA-Z]{3,}", text)
    words = [word for word in words if word not in STOPWORDS]
    return pd.DataFrame(Counter(words).most_common(top_n), columns=["Keyword", "Count"])


def get_api_key() -> str:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return st.session_state.get("manual_api_key", "")


def get_model() -> str:
    try:
        secret_model = st.secrets.get("OPENAI_MODEL", "gpt-5.5")
    except Exception:
        secret_model = "gpt-5.5"
    return st.session_state.get("selected_model", secret_model) or secret_model


def ai_generate(title: str, results: str, context: str = "") -> str:
    api_key = get_api_key()
    if not api_key:
        return "Add your OpenAI API key in the sidebar first."
    if OpenAI is None:
        return "The OpenAI package is not installed."
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are an academic survey data analyst.
Write a concise, credible interpretation of the results below.

Rules:
- Do not invent numbers.
- Use only the provided results.
- Mention sample size limitations when relevant.
- Use formal academic English.
- Separate findings from limitations.
- Do not claim causality or statistical significance unless valid tests are provided.

Title:
{title}

Results:
{results}

Context:
{context}
"""
    try:
        response = client.responses.create(model=get_model(), input=prompt)
        return response.output_text
    except Exception as error:
        return f"AI analysis failed: {error}"


def metric_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">✨ Multi-dataset analytics workspace</div>
            <div class="hero-title">Survey Analytics <span class="hero-gradient">Studio</span></div>
            <div class="hero-subtitle">
                Upload CSV or Excel datasets, explore patterns, review data quality,
                test reliability, inspect open-ended responses, compare files, and generate AI-assisted interpretations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_center():
    st.markdown(
        """
        <div class="upload-shell">
            <div class="upload-title">Upload your datasets</div>
            <div class="upload-subtitle">
                Drag and drop one or more files here. The dashboard will automatically create a tab for every uploaded file.
                The Comparison tab appears only when you upload two or more datasets.
            </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported formats: CSV, XLSX, XLS. You can upload multiple files at once.",
    )
    st.markdown(
        """
            <div class="upload-features">
                <div class="upload-feature">📁 Dynamic file tabs</div>
                <div class="upload-feature">⚖️ Auto comparison</div>
                <div class="upload-feature">🤖 AI interpretation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return uploaded_files


def render_project_overview(datasets: dict):
    st.header("Project Overview")
    total_files = len(datasets)
    total_rows = sum(item["df"].shape[0] for item in datasets.values())
    total_columns = sum(item["df"].shape[1] for item in datasets.values())
    total_missing = sum(int(item["df"].isna().sum().sum()) for item in datasets.values())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Datasets", total_files, "Uploaded files")
    with c2:
        metric_card("Total rows", total_rows, "Across all files")
    with c3:
        metric_card("Total columns", total_columns, "Combined columns")
    with c4:
        metric_card("Missing cells", total_missing, "Across all files")

    rows = []
    for name, item in datasets.items():
        df = item["df"]
        rows.append({
            "Dataset": name,
            "Original file": item["filename"],
            "Size KB": item["size_kb"],
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Numeric columns": len(numeric_columns(df)),
            "Text columns": len(likely_text_columns(df)),
            "Missing cells": int(df.isna().sum().sum()),
        })

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Uploaded files summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def overview_lab(df, name):
    st.markdown(f"### {name} Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", df.shape[0], "Records")
    with c2:
        metric_card("Columns", df.shape[1], "Variables")
    with c3:
        metric_card("Numeric", len(numeric_columns(df)), "Potential scale items")
    with c4:
        metric_card("Missing", int(df.isna().sum().sum()), "Blank cells")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Preview")
    st.dataframe(df.head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def quality_lab(df, name):
    st.markdown(f"### {name} Data Quality")
    numeric_cols = numeric_columns(df)
    quality = pd.DataFrame({
        "Column": df.columns,
        "Detected type": ["Numeric" if col in numeric_cols else "Text / Categorical" for col in df.columns],
        "Missing count": df.isna().sum().values,
        "Missing %": (df.isna().sum().values / max(len(df), 1) * 100).round(1),
        "Unique values": [df[col].nunique(dropna=True) for col in df.columns],
    })
    st.dataframe(quality, use_container_width=True)
    if not quality.empty:
        fig = px.bar(
            quality.sort_values("Missing %", ascending=False).head(20),
            x="Column",
            y="Missing %",
            text="Missing %",
            title="Top missingness by column",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)


def questions_lab(df, name):
    st.markdown(f"### {name} Columns / Questions")
    rows = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(3).tolist()
        rows.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-missing": int(df[col].notna().sum()),
            "Unique": int(df[col].nunique(dropna=True)),
            "Example values": " | ".join(sample),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def charts_lab(df, name):
    st.markdown(f"### {name} Charts")
    chartable_cols = categorical_columns(df, max_unique=60) + [col for col in numeric_columns(df) if col not in categorical_columns(df, max_unique=60)]
    if not chartable_cols:
        st.info("No suitable columns found for charts.")
        return
    selected_col = st.selectbox("Choose a column", chartable_cols, key=f"{name}_chart_col")
    table = summary_table(df, selected_col).head(30)
    st.dataframe(table, use_container_width=True)
    fig = px.bar(table, x="Response", y="Count", text="Percentage", title=f"Distribution: {selected_col}")
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


def filters_lab(df, name):
    st.markdown(f"### {name} Filters")
    if df.empty:
        st.info("Dataset is empty.")
        return
    filter_col = st.selectbox("Filter column", list(df.columns), key=f"{name}_filter_col")
    values = sorted(df[filter_col].dropna().astype(str).unique().tolist())
    selected_values = st.multiselect("Select values", values, key=f"{name}_filter_values")
    filtered_df = df.copy()
    if selected_values:
        filtered_df = filtered_df[filtered_df[filter_col].astype(str).isin(selected_values)]
    st.write(f"Filtered rows: **{len(filtered_df)}** out of **{len(df)}**")
    st.dataframe(filtered_df.head(100), use_container_width=True)


def relationships_lab(df, name):
    st.markdown(f"### {name} Relationships")
    numeric_cols = numeric_columns(df)
    cat_cols = categorical_columns(df, max_unique=40)
    relationship_type = st.selectbox(
        "Relationship type",
        ["Numeric vs Numeric", "Categorical crosstab", "Numeric by Category"],
        key=f"{name}_relationship_type",
    )

    if relationship_type == "Numeric vs Numeric":
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns.")
            return
        x_col = st.selectbox("X-axis", numeric_cols, key=f"{name}_rel_x")
        y_options = [col for col in numeric_cols if col != x_col]
        y_col = st.selectbox("Y-axis", y_options, key=f"{name}_rel_y")
        chart_df = df[[x_col, y_col]].copy()
        chart_df[x_col] = coerce_numeric_series(chart_df[x_col])
        chart_df[y_col] = coerce_numeric_series(chart_df[y_col])
        chart_df = chart_df.dropna()
        if chart_df.empty:
            st.info("No valid numeric rows after cleaning.")
            return
        fig = px.scatter(chart_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)
        if len(chart_df) >= 3:
            corr = chart_df[x_col].corr(chart_df[y_col])
            if pd.notna(corr):
                st.metric("Pearson correlation", round(float(corr), 3))

    elif relationship_type == "Categorical crosstab":
        if len(cat_cols) < 2:
            st.info("Need at least two categorical columns.")
            return
        row_col = st.selectbox("Rows", cat_cols, key=f"{name}_cross_row")
        col_options = [col for col in cat_cols if col != row_col]
        col_col = st.selectbox("Columns", col_options, key=f"{name}_cross_col")
        table = pd.crosstab(df[row_col].fillna("Missing"), df[col_col].fillna("Missing"))
        st.dataframe(table, use_container_width=True)
        fig = px.imshow(table, text_auto=True, title=f"{row_col} by {col_col}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        if not numeric_cols or not cat_cols:
            st.info("Need at least one numeric column and one categorical column.")
            return
        num_col = st.selectbox("Numeric column", numeric_cols, key=f"{name}_num_by_cat_num")
        cat_col = st.selectbox("Category column", cat_cols, key=f"{name}_num_by_cat_cat")
        chart_df = df[[num_col, cat_col]].copy()
        chart_df[num_col] = coerce_numeric_series(chart_df[num_col])
        chart_df = chart_df.dropna()
        if chart_df.empty:
            st.info("No valid rows after cleaning.")
            return
        fig = px.box(chart_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)


def reliability_lab(df, name):
    st.markdown(f"### {name} Reliability Analysis")
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for Cronbach’s alpha.")
        return
    selected_cols = st.multiselect(
        "Select numeric columns",
        numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))],
        key=f"{name}_alpha_cols",
    )
    if len(selected_cols) < 2:
        st.info("Select at least two numeric columns.")
        return
    alpha, valid_rows, item_count = cronbach_alpha(df[selected_cols])
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Cronbach’s alpha", "N/A" if alpha is None else round(alpha, 3), alpha_label(alpha))
    with c2:
        metric_card("Valid rows", valid_rows, "Rows after removing missing values")
    with c3:
        metric_card("Items", item_count, "Selected columns")
    if alpha is not None:
        corr = df[selected_cols].apply(pd.to_numeric, errors="coerce").corr()
        fig = px.imshow(corr, text_auto=True, title="Inter-item correlation matrix")
        st.plotly_chart(fig, use_container_width=True)


def text_lab(df, name):
    st.markdown(f"### {name} Text Analysis")
    text_cols = likely_text_columns(df)
    if not text_cols:
        st.info("No likely open-text columns found.")
        return
    selected_col = st.selectbox("Open-text column", text_cols, key=f"{name}_text_col")
    keywords = text_keywords(df[selected_col], top_n=25)
    if keywords.empty:
        st.info("No keywords found.")
        return
    c1, c2 = st.columns([0.52, 0.48])
    with c1:
        st.dataframe(keywords, use_container_width=True)
    with c2:
        fig = px.bar(keywords.head(15), x="Count", y="Keyword", orientation="h", title="Top keywords")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Sample responses")
    st.dataframe(df[[selected_col]].dropna().head(25), use_container_width=True)


def ai_lab(df, name):
    st.markdown(f"### {name} AI Interpretation")
    st.write("Generate an AI interpretation from a selected column summary. The API key is entered in the sidebar and is not hard-coded.")
    cols = list(df.columns)
    if not cols:
        st.info("Dataset has no columns.")
        return
    selected_col = st.selectbox("Column to interpret", cols, key=f"{name}_ai_col")
    table = summary_table(df, selected_col).head(25)
    st.dataframe(table, use_container_width=True)
    if st.button("Generate AI interpretation", key=f"{name}_ai_button"):
        results = table.to_string(index=False)
        with st.spinner("Generating interpretation..."):
            st.write(ai_generate(f"{name} - {selected_col}", results, f"Rows: {len(df)}"))


def render_dataset_tab(dataset_name: str, df: pd.DataFrame):
    subtabs = st.tabs(["Overview", "Quality", "Questions", "Charts", "Filters", "Relationships", "Reliability", "Text", "AI"])
    with subtabs[0]:
        overview_lab(df, dataset_name)
    with subtabs[1]:
        quality_lab(df, dataset_name)
    with subtabs[2]:
        questions_lab(df, dataset_name)
    with subtabs[3]:
        charts_lab(df, dataset_name)
    with subtabs[4]:
        filters_lab(df, dataset_name)
    with subtabs[5]:
        relationships_lab(df, dataset_name)
    with subtabs[6]:
        reliability_lab(df, dataset_name)
    with subtabs[7]:
        text_lab(df, dataset_name)
    with subtabs[8]:
        ai_lab(df, dataset_name)


def comparison_tab(datasets: dict):
    st.header("Dataset Comparison")
    names = list(datasets.keys())
    if len(names) < 2:
        st.info("Upload at least two datasets to enable comparison.")
        return
    c1, c2 = st.columns(2)
    dataset_a = c1.selectbox("Dataset A", names, index=0)
    dataset_b = c2.selectbox("Dataset B", names, index=1 if len(names) > 1 else 0)
    df_a = datasets[dataset_a]["df"]
    df_b = datasets[dataset_b]["df"]
    summary = pd.DataFrame([
        {"Dataset": dataset_a, "Rows": df_a.shape[0], "Columns": df_a.shape[1], "Missing cells": int(df_a.isna().sum().sum())},
        {"Dataset": dataset_b, "Rows": df_b.shape[0], "Columns": df_b.shape[1], "Missing cells": int(df_b.isna().sum().sum())},
    ])
    st.dataframe(summary, use_container_width=True)
    common_numeric = sorted(set(numeric_columns(df_a)).intersection(set(numeric_columns(df_b))))
    if common_numeric:
        metric = st.selectbox("Common numeric column", common_numeric)
        compare_df = pd.DataFrame({
            "Dataset": [dataset_a, dataset_b],
            "Mean": [coerce_numeric_series(df_a[metric]).mean(), coerce_numeric_series(df_b[metric]).mean()],
            "Median": [coerce_numeric_series(df_a[metric]).median(), coerce_numeric_series(df_b[metric]).median()],
        })
        st.dataframe(compare_df, use_container_width=True)
        fig = px.bar(compare_df, x="Dataset", y="Mean", title=f"Mean comparison: {metric}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common numeric columns found between the selected datasets.")


def export_tab(datasets: dict):
    st.header("Export")
    summary_rows = []
    for name, item in datasets.items():
        df = item["df"]
        summary_rows.append({
            "Dataset": name,
            "Original file": item["filename"],
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing cells": int(df.isna().sum().sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        "Download summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="survey_analytics_summary.csv",
        mime="text/csv",
    )
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for name, item in datasets.items():
            safe_sheet = re.sub(r"[^A-Za-z0-9 _-]", "", name)[:31] or "Dataset"
            item["df"].head(5000).to_excel(writer, sheet_name=safe_sheet, index=False)
    st.download_button(
        "Download Excel workbook",
        data=excel_buffer.getvalue(),
        file_name="survey_analytics_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Sidebar: settings only
st.sidebar.title("Settings")
st.sidebar.write("Configure AI analysis. File upload is now in the main workspace.")
st.sidebar.markdown("---")
st.sidebar.subheader("AI Settings")
st.session_state["manual_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get("manual_api_key", ""))
st.session_state["selected_model"] = st.sidebar.text_input("OpenAI Model", value=st.session_state.get("selected_model", "gpt-5.5"))
st.sidebar.caption("Each user can enter their own API key. Do not hard-code API keys inside the app.")

# Main app
render_hero()
uploaded_files = render_upload_center()

datasets = build_datasets(uploaded_files) if uploaded_files else {}

if not datasets:
    st.markdown(
        """
        <div class="soft-note">
            <b>How this studio works:</b><br>
            • Upload one file to analyze that dataset only.<br>
            • Upload two or more files to unlock the Comparison tab.<br>
            • Tab names are generated from the uploaded file names.<br>
            • No fixed labels like Lecturer or Student are used anymore.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown("### Loaded datasets")
for dataset_name, item in datasets.items():
    st.markdown(
        f"<div class='dataset-pill'>📁 {dataset_name} <span style='color:#94a3b8;'>— {item['filename']} · {item['size_kb']} KB</span></div>",
        unsafe_allow_html=True,
    )

tab_labels = ["🏠 Overview"] + [f"📁 {name}" for name in datasets.keys()]
if len(datasets) > 1:
    tab_labels.append("⚖️ Comparison")
tab_labels.append("📦 Export")

tabs = st.tabs(tab_labels)
index = 0
with tabs[index]:
    render_project_overview(datasets)
index += 1

for dataset_name, item in datasets.items():
    with tabs[index]:
        render_dataset_tab(dataset_name, item["df"])
    index += 1

if len(datasets) > 1:
    with tabs[index]:
        comparison_tab(datasets)
    index += 1

with tabs[index]:
    export_tab(datasets)
