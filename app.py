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
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 35%, #0b1120 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label, span, div {
        color: #e5e7eb;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937;
        border-radius: 14px;
        padding: 12px 18px;
        color: #e5e7eb;
        border: 1px solid #334155;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #111827);
        border: 1px solid #334155;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 14px 35px rgba(0,0,0,0.25);
        min-height: 145px;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        word-break: break-word;
    }

    .metric-sub {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 6px;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid #334155;
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .hero {
        background: radial-gradient(circle at top left, rgba(37,99,235,.35), transparent 35%),
                    linear-gradient(135deg, #111827 0%, #1e1b4b 55%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 28px;
        padding: 34px;
        margin-bottom: 24px;
        box-shadow: 0 18px 45px rgba(0,0,0,.28);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.05;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 950px;
        margin-top: 12px;
    }

    .file-pill {
        display: inline-block;
        margin: 4px 6px 4px 0;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(37,99,235,.18);
        border: 1px solid rgba(96,165,250,.35);
        color: #dbeafe;
        font-size: .85rem;
        font-weight: 700;
    }

    .note-box {
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid #334155;
        background: rgba(30, 41, 59, 0.75);
        color: #cbd5e1;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


STOPWORDS = set("""
the and to of in a an is are was were for with on at by from this that these those it as be or if yes no not i you we they he she them their our your about into can could should would there here very more most less also have has had do does did because than then so such using use used survey surveys questionnaire questionnaires response responses data dataset file files analysis answer answers question questions
""".split())


# =========================================================
# Data Loading and Cleaning
# =========================================================

def make_unique_columns(columns):
    cleaned_columns = (
        pd.Series(columns)
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
        .replace("", "Unnamed")
        .tolist()
    )

    seen = {}
    unique_columns = []

    for column in cleaned_columns:
        if column not in seen:
            seen[column] = 1
            unique_columns.append(column)
        else:
            seen[column] += 1
            unique_columns.append(f"{column}_{seen[column]}")

    return unique_columns


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    return df


def clean_dataset_name(filename: str) -> str:
    name = Path(filename).stem
    name = name.replace("_", " ").replace("-", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name.title() or "Dataset"


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")

    return clean_dataframe(df)


def build_datasets(uploaded_files):
    datasets = {}

    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.getvalue()
            df = load_uploaded_file(file_bytes, uploaded_file.name)
            dataset_name = clean_dataset_name(uploaded_file.name)

            base_name = dataset_name
            counter = 2
            while dataset_name in datasets:
                dataset_name = f"{base_name} ({counter})"
                counter += 1

            datasets[dataset_name] = {
                "filename": uploaded_file.name,
                "df": df,
                "size": len(file_bytes),
            }

        except Exception as error:
            st.sidebar.error(f"Failed to load {uploaded_file.name}: {error}")

    return datasets


def numeric_columns(df: pd.DataFrame):
    cols = []

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() >= 2:
            cols.append(col)

    return cols


def likely_text_columns(df: pd.DataFrame):
    cols = []

    for col in df.columns:
        s = df[col].dropna().astype(str)

        if len(s) == 0:
            continue

        avg_len = s.str.len().mean()
        unique_ratio = s.nunique() / max(len(s), 1)

        if avg_len > 25 or unique_ratio > 0.65:
            cols.append(col)

    return cols


def categorical_columns(df: pd.DataFrame, max_unique=30):
    cols = []

    for col in df.columns:
        unique_count = df[col].dropna().astype(str).nunique()

        if 2 <= unique_count <= max_unique:
            cols.append(col)

    return cols


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def split_cell_values(value, separator=";"):
    if pd.isna(value):
        return []

    parts = str(value).split(separator)
    cleaned = []

    for part in parts:
        item = part.strip()

        if item != "" and item.lower() != "nan":
            cleaned.append(item)

    return cleaned


def summary_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df[column].fillna("Missing").astype(str).str.strip()
    counts = data.value_counts(dropna=False)

    table = pd.DataFrame({
        "Response": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / max(len(df), 1) * 100).round(1),
    })

    table["Label"] = (
        table["Count"].astype(str)
        + " ("
        + table["Percentage"].astype(str)
        + "%)"
    )

    return table


# =========================================================
# Reliability and Text
# =========================================================

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

    return pd.DataFrame(
        Counter(words).most_common(top_n),
        columns=["Keyword", "Count"]
    )


# =========================================================
# AI
# =========================================================

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
        return "The OpenAI package is not installed. Run: pip install openai"

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an academic and business survey data analyst.
Write a concise, credible interpretation of the results below.

Strict rules:
- Do not invent numbers.
- Use only the provided results.
- Mention sample size limitations when relevant.
- Use clear professional English.
- Separate findings from limitations.
- Do not overclaim causality.
- Do not claim statistical significance unless a valid test is provided.

Title:
{title}

Results:
{results}

Context:
{context}
"""

    try:
        response = client.responses.create(
            model=get_model(),
            input=prompt
        )
        return response.output_text

    except Exception as error:
        return f"AI analysis failed: {error}"


# =========================================================
# UI Components
# =========================================================

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


def hero_section():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Survey Analytics Studio</div>
            <div class="hero-sub">
                Upload one or more CSV or Excel datasets, explore patterns, review data quality,
                test reliability, inspect open-ended responses, compare datasets, and generate AI-assisted interpretations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_overview(df, name):
    st.markdown(f"### {name} Executive Overview")

    c1, c2, c3, c4 = st.columns(4)

    missing = int(df.isna().sum().sum())
    numeric_count = len(numeric_columns(df))
    text_count = len(likely_text_columns(df))

    with c1:
        metric_card("Rows", df.shape[0], "Total records")

    with c2:
        metric_card("Columns", df.shape[1], "Available fields")

    with c3:
        metric_card("Numeric fields", numeric_count, "Potential scales / metrics")

    with c4:
        metric_card("Missing cells", missing, "Blank or unavailable values")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Detected Structure")
    c1, c2, c3 = st.columns(3)
    c1.metric("Text-like fields", text_count)
    c2.metric("Categorical fields", len(categorical_columns(df)))
    c3.metric("Duplicate rows", int(df.duplicated().sum()))
    st.markdown('</div>', unsafe_allow_html=True)


def data_quality(df, name):
    st.markdown(f"### {name} Data Quality Audit")

    numeric_cols = numeric_columns(df)

    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().sum().values / max(len(df), 1) * 100).round(1),
        "Unique Values": [df[column].nunique(dropna=True) for column in df.columns],
        "Detected Type": [
            "Numeric" if column in numeric_cols else "Text/Categorical"
            for column in df.columns
        ],
    })

    st.dataframe(quality, use_container_width=True)

    fig = px.bar(
        quality.sort_values("Missing %", ascending=False).head(20),
        x="Column",
        y="Missing %",
        text="Missing %",
        title="Top Missingness by Column"
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def questions_lab(df, name):
    st.markdown(f"### {name} Questions / Fields Explorer")

    selected_column = st.selectbox(
        "Choose a column",
        df.columns,
        key=f"{name}_questions_column"
    )

    st.write("**Column type:**", str(df[selected_column].dtype))
    st.write("**Non-missing values:**", int(df[selected_column].notna().sum()))
    st.write("**Unique values:**", int(df[selected_column].nunique(dropna=True)))

    table = summary_table(df, selected_column)

    st.dataframe(table.head(100), use_container_width=True)

    if len(table) <= 30:
        fig = px.bar(
            table,
            x="Response",
            y="Count",
            text="Label",
            title=f"Response Distribution — {selected_column}"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This field has many unique values, so a distribution chart may not be useful.")


def charts_lab(df, name):
    st.markdown(f"### {name} Multi Chart Lab")

    chart_type = st.selectbox(
        "Chart type",
        [
            "Categorical bar chart",
            "Numeric histogram",
            "Pie chart",
            "Box plot",
        ],
        key=f"{name}_chart_type"
    )

    numeric_cols = numeric_columns(df)
    cat_cols = categorical_columns(df, max_unique=60)

    if chart_type == "Categorical bar chart":
        if not cat_cols:
            st.info("No suitable categorical columns found.")
            return

        column = st.selectbox("Column", cat_cols, key=f"{name}_cat_bar")
        table = summary_table(df, column).head(60)

        fig = px.bar(
            table,
            x="Response",
            y="Count",
            text="Label",
            title=f"{column} Distribution"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Numeric histogram":
        if not numeric_cols:
            st.info("No numeric columns found.")
            return

        column = st.selectbox("Numeric column", numeric_cols, key=f"{name}_hist")
        values = coerce_numeric_series(df[column]).dropna()

        fig = px.histogram(values, x=column, title=f"{column} Histogram")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie chart":
        if not cat_cols:
            st.info("No suitable categorical columns found.")
            return

        column = st.selectbox("Column", cat_cols, key=f"{name}_pie")
        table = summary_table(df, column).head(12)

        fig = px.pie(
            table,
            names="Response",
            values="Count",
            title=f"{column} Pie Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box plot":
        if not numeric_cols:
            st.info("No numeric columns found.")
            return

        y_col = st.selectbox("Numeric column", numeric_cols, key=f"{name}_box_y")
        group_col = st.selectbox(
            "Optional grouping column",
            ["None"] + cat_cols,
            key=f"{name}_box_group"
        )

        chart_df = df.copy()
        chart_df[y_col] = coerce_numeric_series(chart_df[y_col])

        if group_col == "None":
            fig = px.box(chart_df, y=y_col, title=f"{y_col} Box Plot")
        else:
            fig = px.box(chart_df, x=group_col, y=y_col, title=f"{y_col} by {group_col}")

        st.plotly_chart(fig, use_container_width=True)


def filters_lab(df, name):
    st.markdown(f"### {name} Filter Lab")

    filter_col = st.selectbox(
        "Filter column",
        df.columns,
        key=f"{name}_filter_col"
    )

    values = df[filter_col].dropna().astype(str).unique().tolist()

    selected_values = st.multiselect(
        "Filter values",
        sorted(values),
        key=f"{name}_filter_values"
    )

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
        [
            "Numeric vs Numeric",
            "Categorical crosstab",
            "Numeric by Category",
        ],
        key=f"{name}_relationship_type"
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

        fig = px.scatter(chart_df, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)

        if len(chart_df) >= 3:
            corr = chart_df[x_col].corr(chart_df[y_col])
            st.metric("Pearson correlation", round(corr, 3))

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

    elif relationship_type == "Numeric by Category":
        if not numeric_cols or not cat_cols:
            st.info("Need at least one numeric column and one categorical column.")
            return

        num_col = st.selectbox("Numeric column", numeric_cols, key=f"{name}_num_by_cat_num")
        cat_col = st.selectbox("Category column", cat_cols, key=f"{name}_num_by_cat_cat")

        chart_df = df[[num_col, cat_col]].copy()
        chart_df[num_col] = coerce_numeric_series(chart_df[num_col])
        chart_df = chart_df.dropna()

        fig = px.box(chart_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)


def reliability_lab(df, name):
    st.markdown(f"### {name} Reliability Analysis")

    numeric_cols = numeric_columns(df)

    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for Cronbach’s alpha.")
        return

    selected_cols = st.multiselect(
        "Select numeric columns for reliability analysis",
        numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))],
        key=f"{name}_alpha_cols"
    )

    if len(selected_cols) < 2:
        st.info("Select at least two numeric columns.")
        return

    alpha, valid_rows, item_count = cronbach_alpha(df[selected_cols])

    c1, c2, c3 = st.columns(3)
    c1.metric("Cronbach’s Alpha", "N/A" if alpha is None else round(alpha, 3))
    c2.metric("Valid rows", valid_rows)
    c3.metric("Items", item_count)

    st.write("**Interpretation:**", alpha_label(alpha))

    corr_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").corr()

    fig = px.imshow(
        corr_df,
        text_auto=True,
        title="Inter-item Correlation Heatmap",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)


def text_lab(df, name):
    st.markdown(f"### {name} Text Analysis")

    text_cols = likely_text_columns(df)

    if not text_cols:
        st.info("No likely open-text columns found. You can still choose from all columns below.")
        text_cols = list(df.columns)

    selected_col = st.selectbox(
        "Select text column",
        text_cols,
        key=f"{name}_text_col"
    )

    keywords = text_keywords(df[selected_col], top_n=25)

    if keywords.empty:
        st.info("No keywords found.")
    else:
        st.dataframe(keywords, use_container_width=True)
        fig = px.bar(
            keywords.head(20),
            x="Keyword",
            y="Count",
            text="Count",
            title=f"Top Keywords — {selected_col}"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    sample_responses = df[selected_col].dropna().astype(str).head(30)

    st.markdown("#### Sample responses")
    st.dataframe(pd.DataFrame({"Response": sample_responses}), use_container_width=True)

    if st.button("Generate AI interpretation", key=f"{name}_text_ai"):
        result_text = keywords.to_string(index=False) if not keywords.empty else "No keywords found."
        context = "\n".join(sample_responses.tolist()[:20])
        st.write(ai_generate(f"Text analysis for {name} - {selected_col}", result_text, context))


def ai_interpretation_lab(df, name):
    st.markdown(f"### {name} AI Interpretation")

    st.write(
        "Generate a high-level interpretation from the selected dataset profile. "
        "The app only sends summary statistics and selected previews, not the full file."
    )

    numeric_cols = numeric_columns(df)
    cat_cols = categorical_columns(df, max_unique=20)
    text_cols = likely_text_columns(df)

    selected_context_cols = st.multiselect(
        "Optional columns to include in the AI context",
        list(df.columns),
        default=list(df.columns[:min(5, len(df.columns))]),
        key=f"{name}_ai_context_cols"
    )

    profile = {
        "dataset": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": numeric_cols[:25],
        "categorical_columns": cat_cols[:25],
        "text_columns": text_cols[:25],
        "missing_cells": int(df.isna().sum().sum()),
    }

    preview = df[selected_context_cols].head(10).to_string(index=False) if selected_context_cols else ""

    st.json(profile)

    if st.button("Generate AI dataset interpretation", key=f"{name}_ai_profile"):
        st.write(
            ai_generate(
                f"Dataset interpretation for {name}",
                str(profile),
                preview
            )
        )


def render_dataset_tab(dataset_name: str, df: pd.DataFrame):
    subtabs = st.tabs([
        "Overview",
        "Quality",
        "Questions",
        "Charts",
        "Filters",
        "Relationships",
        "Reliability",
        "Text",
        "AI",
    ])

    with subtabs[0]:
        dataset_overview(df, dataset_name)

    with subtabs[1]:
        data_quality(df, dataset_name)

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
        ai_interpretation_lab(df, dataset_name)


def render_project_overview(datasets: dict):
    st.header("Project Overview")

    total_files = len(datasets)
    total_rows = sum(item["df"].shape[0] for item in datasets.values())
    total_cols = sum(item["df"].shape[1] for item in datasets.values())
    total_missing = sum(int(item["df"].isna().sum().sum()) for item in datasets.values())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Datasets", total_files, "Uploaded files")

    with c2:
        metric_card("Total rows", total_rows, "Combined records")

    with c3:
        metric_card("Total columns", total_cols, "Across all files")

    with c4:
        metric_card("Missing cells", total_missing, "Across all files")

    summary_rows = []

    for name, item in datasets.items():
        df = item["df"]
        summary_rows.append({
            "Dataset": name,
            "Original File": item["filename"],
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Numeric Fields": len(numeric_columns(df)),
            "Text Fields": len(likely_text_columns(df)),
            "Missing Cells": int(df.isna().sum().sum()),
        })

    st.markdown("### Uploaded Files Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown("### Loaded dataset tabs")
    pills = "".join([f'<span class="file-pill">📁 {name}</span>' for name in datasets.keys()])
    st.markdown(pills, unsafe_allow_html=True)


def render_comparison_tab(datasets: dict):
    st.header("Dataset Comparison")

    dataset_names = list(datasets.keys())

    if len(dataset_names) < 2:
        st.info("Upload at least two datasets to enable comparison.")
        return

    c1, c2 = st.columns(2)
    dataset_a_name = c1.selectbox("Dataset A", dataset_names, index=0)
    dataset_b_name = c2.selectbox("Dataset B", dataset_names, index=1)

    df_a = datasets[dataset_a_name]["df"]
    df_b = datasets[dataset_b_name]["df"]

    compare_rows = pd.DataFrame([
        {
            "Dataset": dataset_a_name,
            "Rows": df_a.shape[0],
            "Columns": df_a.shape[1],
            "Numeric Fields": len(numeric_columns(df_a)),
            "Text Fields": len(likely_text_columns(df_a)),
            "Missing Cells": int(df_a.isna().sum().sum()),
        },
        {
            "Dataset": dataset_b_name,
            "Rows": df_b.shape[0],
            "Columns": df_b.shape[1],
            "Numeric Fields": len(numeric_columns(df_b)),
            "Text Fields": len(likely_text_columns(df_b)),
            "Missing Cells": int(df_b.isna().sum().sum()),
        },
    ])

    st.dataframe(compare_rows, use_container_width=True)

    st.markdown("### Shared columns")
    common_cols = sorted(set(df_a.columns).intersection(set(df_b.columns)))
    st.write(f"Shared columns: **{len(common_cols)}**")

    if common_cols:
        st.dataframe(pd.DataFrame({"Shared Column": common_cols}), use_container_width=True)

    common_numeric = sorted(set(numeric_columns(df_a)).intersection(set(numeric_columns(df_b))))

    if common_numeric:
        selected_metric = st.selectbox("Compare numeric column", common_numeric)

        values_a = coerce_numeric_series(df_a[selected_metric]).dropna()
        values_b = coerce_numeric_series(df_b[selected_metric]).dropna()

        compare_chart = pd.DataFrame({
            "Dataset": [dataset_a_name, dataset_b_name],
            "Mean": [values_a.mean(), values_b.mean()],
            "Median": [values_a.median(), values_b.median()],
            "Valid N": [len(values_a), len(values_b)],
        })

        st.markdown("### Numeric comparison")
        st.dataframe(compare_chart, use_container_width=True)

        fig = px.bar(compare_chart, x="Dataset", y="Mean", text="Mean", title=f"Mean Comparison — {selected_metric}")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common numeric columns found between the selected datasets.")

    if st.button("Generate AI comparison interpretation"):
        st.write(
            ai_generate(
                f"Comparison: {dataset_a_name} vs {dataset_b_name}",
                compare_rows.to_string(index=False),
                f"Common columns: {common_cols[:30]}"
            )
        )


def render_export_tab(datasets: dict):
    st.header("Export")

    export_summary = []

    for name, item in datasets.items():
        export_summary.append({
            "Dataset": name,
            "Original File": item["filename"],
            "Rows": item["df"].shape[0],
            "Columns": item["df"].shape[1],
            "Missing Cells": int(item["df"].isna().sum().sum()),
        })

    export_df = pd.DataFrame(export_summary)
    st.dataframe(export_df, use_container_width=True)

    csv = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download project summary CSV",
        data=csv,
        file_name="survey_analytics_project_summary.csv",
        mime="text/csv",
    )

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Project Summary")

        for name, item in datasets.items():
            safe_sheet_name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
            item["df"].head(5000).to_excel(writer, index=False, sheet_name=safe_sheet_name)

    st.download_button(
        "Download Excel workbook",
        data=output.getvalue(),
        file_name="survey_analytics_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("Control Panel")
st.sidebar.write("Upload one or more datasets and configure analysis.")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV or Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Upload one file for a single-dataset analysis, or multiple files to enable comparison.",
)

datasets = build_datasets(uploaded_files) if uploaded_files else {}

if datasets:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Loaded Datasets")

    for name, item in datasets.items():
        st.sidebar.success(name)
        st.sidebar.caption(item["filename"])

st.sidebar.markdown("---")
st.sidebar.subheader("AI Settings")
st.session_state["manual_api_key"] = st.sidebar.text_input(
    "OpenAI API Key",
    value=st.session_state.get("manual_api_key", ""),
    type="password",
)
st.session_state["selected_model"] = st.sidebar.text_input(
    "OpenAI Model",
    value=st.session_state.get("selected_model", "gpt-5.5"),
)

st.sidebar.caption("Each user can enter their own API key. Do not hard-code API keys inside the app.")


# =========================================================
# Main UI
# =========================================================

hero_section()

if not datasets:
    st.info("Upload at least one CSV or Excel file from the sidebar to begin.")

    st.markdown(
        """
        <div class="note-box">
            <b>How this version works:</b><br>
            • Upload one file to analyze that dataset only.<br>
            • Upload two or more files to unlock the Comparison tab.<br>
            • Tab names are generated from the uploaded file names.<br>
            • No fixed labels like Lecturer or Student are used anymore.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

tab_labels = ["🏠 Overview"]
tab_labels += [f"📁 {name}" for name in datasets.keys()]

if len(datasets) > 1:
    tab_labels += ["⚖️ Comparison"]

tab_labels += ["📦 Export"]

tabs = st.tabs(tab_labels)

tab_index = 0

with tabs[tab_index]:
    render_project_overview(datasets)
tab_index += 1

for dataset_name, item in datasets.items():
    with tabs[tab_index]:
        render_dataset_tab(dataset_name, item["df"])
    tab_index += 1

if len(datasets) > 1:
    with tabs[tab_index]:
        render_comparison_tab(datasets)
    tab_index += 1

with tabs[tab_index]:
    render_export_tab(datasets)
