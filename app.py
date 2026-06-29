import io
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(
    page_title="Pro AI Survey Analytics",
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
        max-width: 900px;
        margin-top: 12px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


STOPWORDS = set("""
the and to of in a an is are was were for with on at by from this that these those it as be or if yes no not i you we they he she them their our your about into can could should would there here very more most less also have has had do does did because than then so such using use used student students lecturer lecturers assessment speaking course feedback formative summative learning teaching
""".split())


# =========================================================
# Data Cleaning Functions
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


def split_summary_table(df: pd.DataFrame, column: str, separator=";") -> pd.DataFrame:
    answers = []

    for value in df[column].dropna():
        answers.extend(split_cell_values(value, separator))

    if len(answers) == 0:
        return pd.DataFrame(columns=["Response", "Count", "Percentage", "Label"])

    counts = pd.Series(answers).value_counts().reset_index()
    counts.columns = ["Response", "Count"]

    total = counts["Count"].sum()

    counts["Percentage"] = (
        counts["Count"] / max(total, 1) * 100
    ).round(1)

    counts["Label"] = (
        counts["Count"].astype(str)
        + " ("
        + counts["Percentage"].astype(str)
        + "%)"
    )

    return counts


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


def create_heatmap_with_labels(table, title):
    total = table.values.sum()

    if total == 0:
        total = 1

    percentages = (table / total * 100).round(1)

    text_labels = table.astype(str) + "<br>(" + percentages.astype(str) + "%)"

    fig = go.Figure(
        data=go.Heatmap(
            z=table.values,
            x=table.columns.astype(str),
            y=table.index.astype(str),
            text=text_labels.values,
            texttemplate="%{text}",
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Count: %{z}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Column Category",
        yaxis_title="Row Category"
    )

    return fig


# =========================================================
# Cronbach Alpha
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


# =========================================================
# AI Functions
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
        return "The OpenAI package is not installed. Run: pip3 install openai"

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an academic survey data analyst.
Write a concise, credible interpretation of the results below.

Strict rules:
- Do not invent numbers.
- Use only the provided results.
- Mention sample size limitations when relevant.
- Use formal academic English.
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
# Text Analysis
# =========================================================

def text_keywords(series: pd.Series, top_n=20):
    text = " ".join(series.dropna().astype(str).tolist()).lower()
    words = re.findall(r"[a-zA-Z]{3,}", text)
    words = [word for word in words if word not in STOPWORDS]

    return pd.DataFrame(
        Counter(words).most_common(top_n),
        columns=["Keyword", "Count"]
    )


# =========================================================
# UI Helpers
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


def dataset_overview(df, name):
    st.markdown(f"### {name} Executive Overview")

    c1, c2, c3, c4 = st.columns(4)

    missing = int(df.isna().sum().sum())
    numeric_count = len(numeric_columns(df))
    text_count = len(likely_text_columns(df))

    with c1:
        metric_card("Responses", df.shape[0], "Total rows")

    with c2:
        metric_card("Questions", df.shape[1], "Total columns")

    with c3:
        metric_card("Numeric items", numeric_count, "Potential Likert / coded items")

    with c4:
        metric_card("Missing cells", missing, "Blank or unavailable values")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Preview")
    st.dataframe(df.head(10), width="stretch")
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

    st.dataframe(quality, width="stretch")

    fig = px.bar(
        quality.sort_values("Missing %", ascending=False).head(20),
        x="Column",
        y="Missing %",
        text="Missing %",
        title="Top Missingness by Column"
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")


# =========================================================
# Analysis Labs
# =========================================================

def question_lab(df, name):
    st.markdown(f"### {name} Question Lab")

    col = st.selectbox(
        "Choose a question",
        df.columns,
        key=f"{name}_ql_col"
    )

    chart = st.radio(
        "Chart type",
        ["Bar", "Pie", "Donut"],
        horizontal=True,
        key=f"{name}_ql_chart"
    )

    split_answers = st.checkbox(
        "Split multiple answers inside cells",
        value=True,
        key=f"{name}_ql_split"
    )

    separator = st.text_input(
        "Separator between answers",
        value=";",
        key=f"{name}_ql_separator"
    )

    if split_answers:
        table = split_summary_table(df, col, separator)
    else:
        table = summary_table(df, col)

    if table.empty:
        st.warning("No valid answers found.")
        return

    st.dataframe(table[["Response", "Count", "Percentage"]], width="stretch")

    if chart == "Bar":
        fig = px.bar(
            table,
            x="Response",
            y="Count",
            text="Label",
            title=col
        )

        fig.update_traces(textposition="outside", cliponaxis=False)

    else:
        hole = 0.45 if chart == "Donut" else 0

        fig = px.pie(
            table,
            names="Response",
            values="Count",
            hole=hole,
            title=col
        )

        fig.update_traces(textinfo="label+percent+value")

    fig.update_layout(
        xaxis_title="Response",
        yaxis_title="Count",
        uniformtext_minsize=9,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, width="stretch")

    if st.button("Generate AI interpretation", key=f"{name}_ql_ai"):
        st.write(
            ai_generate(
                f"{name}: {col}",
                table[["Response", "Count", "Percentage"]].to_string(index=False),
                f"Sample size: {len(df)}"
            )
        )


def multi_variable_chart_lab(df, name):
    st.markdown(f"### {name} Multi-Variable Chart Lab")
    st.write("Choose two or more columns, then select the chart type you want.")

    selected_columns = st.multiselect(
        "Choose two or more columns",
        df.columns,
        key=f"{name}_multi_variable_columns"
    )

    if len(selected_columns) < 2:
        st.info("Please choose at least two columns.")
        return

    chart_type = st.selectbox(
        "Choose chart type",
        [
            "Count comparison",
            "Grouped bar chart",
            "Stacked bar chart",
            "Percentage stacked bar chart",
            "Heatmap between two variables",
            "Box plot for numeric variables",
            "Scatter plot for two numeric variables"
        ],
        key=f"{name}_multi_variable_chart_type"
    )

    split_multiple_answers = st.checkbox(
        "Split multiple answers inside cells",
        value=True,
        key=f"{name}_multi_variable_split"
    )

    separator = st.text_input(
        "Separator used between answers",
        value=";",
        key=f"{name}_multi_variable_separator"
    )

    chart_df = df[selected_columns].copy()

    if split_multiple_answers:
        long_rows = []

        for column in selected_columns:
            for value in chart_df[column].dropna():
                parts = split_cell_values(value, separator)

                for part in parts:
                    long_rows.append({
                        "Question": column,
                        "Answer": part
                    })

        long_df = pd.DataFrame(long_rows)

    else:
        long_df = chart_df.melt(
            var_name="Question",
            value_name="Answer"
        ).dropna()

        long_df["Answer"] = long_df["Answer"].astype(str).str.strip()
        long_df = long_df[long_df["Answer"] != ""]

    if long_df.empty:
        st.warning("No valid data found in the selected columns.")
        return

    if chart_type in [
        "Count comparison",
        "Grouped bar chart",
        "Stacked bar chart",
        "Percentage stacked bar chart"
    ]:
        summary = (
            long_df
            .groupby(["Question", "Answer"])
            .size()
            .reset_index(name="Count")
        )

        summary["Percentage"] = (
            summary.groupby("Question")["Count"]
            .transform(lambda x: x / x.sum() * 100)
            .round(1)
        )

        summary["Label"] = (
            summary["Count"].astype(str)
            + " ("
            + summary["Percentage"].astype(str)
            + "%)"
        )

        st.subheader("Summary table")
        st.dataframe(
            summary[["Question", "Answer", "Count", "Percentage"]],
            width="stretch"
        )

        if chart_type == "Count comparison":
            fig = px.bar(
                summary,
                x="Question",
                y="Count",
                color="Answer",
                text="Label",
                title=f"{name}: Count comparison across selected variables"
            )

            fig.update_traces(textposition="inside")

        elif chart_type == "Grouped bar chart":
            fig = px.bar(
                summary,
                x="Answer",
                y="Count",
                color="Question",
                barmode="group",
                text="Label",
                title=f"{name}: Grouped bar chart"
            )

            fig.update_traces(textposition="outside", cliponaxis=False)

        elif chart_type == "Stacked bar chart":
            fig = px.bar(
                summary,
                x="Question",
                y="Count",
                color="Answer",
                barmode="stack",
                text="Label",
                title=f"{name}: Stacked bar chart"
            )

            fig.update_traces(textposition="inside")

        else:
            fig = px.bar(
                summary,
                x="Question",
                y="Percentage",
                color="Answer",
                barmode="stack",
                text="Label",
                title=f"{name}: Percentage stacked bar chart"
            )

            fig.update_traces(textposition="inside")
            fig.update_yaxes(title="Percentage")

        fig.update_layout(
            xaxis_title="Question / Answer",
            yaxis_title="Count / Percentage",
            uniformtext_minsize=9,
            uniformtext_mode="hide",
            legend_title_text="Answer / Question"
        )

        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Heatmap between two variables":
        row_column = st.selectbox(
            "Choose row variable",
            selected_columns,
            key=f"{name}_heatmap_row"
        )

        column_column = st.selectbox(
            "Choose column variable",
            selected_columns,
            key=f"{name}_heatmap_column"
        )

        if row_column == column_column:
            st.warning("Please choose two different variables.")
            return

        split_heatmap = st.checkbox(
            "Split answers in heatmap variables",
            value=True,
            key=f"{name}_heatmap_split"
        )

        heatmap_rows = []

        for _, record in df[[row_column, column_column]].dropna(how="all").iterrows():
            row_raw = record[row_column]
            col_raw = record[column_column]

            if pd.isna(row_raw) or pd.isna(col_raw):
                continue

            if split_heatmap:
                row_values = split_cell_values(row_raw, separator)
                col_values = split_cell_values(col_raw, separator)
            else:
                row_values = [str(row_raw).strip()]
                col_values = [str(col_raw).strip()]

            for row_value in row_values:
                for col_value in col_values:
                    heatmap_rows.append({
                        "Row": row_value,
                        "Column": col_value
                    })

        heatmap_df = pd.DataFrame(heatmap_rows)

        if heatmap_df.empty:
            st.warning("No valid data found for heatmap.")
            return

        table = pd.crosstab(
            heatmap_df["Row"],
            heatmap_df["Column"]
        )

        st.subheader("Crosstab table")
        st.dataframe(table, width="stretch")

        fig = create_heatmap_with_labels(
            table,
            f"{name}: Heatmap of {row_column} vs {column_column}"
        )

        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Box plot for numeric variables":
        numeric_df = chart_df.apply(pd.to_numeric, errors="coerce")

        long_numeric = numeric_df.melt(
            var_name="Question",
            value_name="Value"
        ).dropna()

        if long_numeric.empty:
            st.warning("No numeric values found in the selected columns.")
            return

        st.subheader("Numeric summary")
        st.dataframe(numeric_df.describe().T, width="stretch")

        fig = px.box(
            long_numeric,
            x="Question",
            y="Value",
            points="all",
            title=f"{name}: Box plot for selected numeric variables"
        )

        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Scatter plot for two numeric variables":
        if len(selected_columns) != 2:
            st.warning("For scatter plot, please choose exactly two columns.")
            return

        x_col = selected_columns[0]
        y_col = selected_columns[1]

        scatter_df = df[[x_col, y_col]].copy()
        scatter_df[x_col] = pd.to_numeric(scatter_df[x_col], errors="coerce")
        scatter_df[y_col] = pd.to_numeric(scatter_df[y_col], errors="coerce")
        scatter_df = scatter_df.dropna()

        if scatter_df.empty:
            st.warning("No numeric values found for scatter plot.")
            return

        fig = px.scatter(
            scatter_df,
            x=x_col,
            y=y_col,
            title=f"{name}: Scatter plot of {x_col} vs {y_col}"
        )

        st.plotly_chart(fig, width="stretch")

    if st.button("Generate AI interpretation for selected variables", key=f"{name}_multi_variable_ai"):
        ai_context = f"""
Dataset: {name}
Selected columns: {selected_columns}
Chart type: {chart_type}

Preview of analysed data:
{long_df.head(200).to_string(index=False)}
"""

        st.write(
            ai_generate(
                f"{name}: Multi-variable chart analysis",
                ai_context,
                "Interpret the selected variables academically. Explain the main patterns, differences, percentages, and limitations."
            )
        )


def filter_lab(df, name):
    st.markdown(f"### {name} Filter Lab")

    c1, c2, c3 = st.columns(3)

    with c1:
        filter_col = st.selectbox(
            "Filter variable",
            df.columns,
            key=f"{name}_filter_col"
        )

    values = ["All"] + sorted(
        df[filter_col].dropna().astype(str).unique().tolist()
    )

    with c2:
        value = st.selectbox(
            "Filter value",
            values,
            key=f"{name}_filter_value"
        )

    with c3:
        analysis_col = st.selectbox(
            "Question to analyse",
            df.columns,
            key=f"{name}_filter_analysis"
        )

    if value == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df[filter_col].astype(str) == str(value)]

    st.info(f"Filtered sample size: {len(filtered_df)}")

    if len(filtered_df) == 0:
        st.warning("No records match this filter.")
        return

    split_answers = st.checkbox(
        "Split multiple answers in analysed question",
        value=True,
        key=f"{name}_filter_split"
    )

    separator = st.text_input(
        "Separator between answers",
        value=";",
        key=f"{name}_filter_separator"
    )

    if split_answers:
        table = split_summary_table(filtered_df, analysis_col, separator)
    else:
        table = summary_table(filtered_df, analysis_col)

    if table.empty:
        st.warning("No valid answers found.")
        return

    st.dataframe(table[["Response", "Count", "Percentage"]], width="stretch")

    fig = px.bar(
        table,
        x="Response",
        y="Count",
        text="Label",
        title=f"{analysis_col} | {filter_col}: {value}"
    )

    fig.update_traces(textposition="outside", cliponaxis=False)

    st.plotly_chart(fig, width="stretch")

    if st.button("Generate AI filtered interpretation", key=f"{name}_filter_ai"):
        st.write(
            ai_generate(
                f"{name}: Filtered Analysis",
                table[["Response", "Count", "Percentage"]].to_string(index=False),
                f"Filter: {filter_col} = {value}. Sample size: {len(filtered_df)}"
            )
        )


def yes_no_pattern_lab(df, name):
    st.markdown(f"### {name} Answer Pattern Lab")

    analysis_type = st.radio(
        "Analysis type",
        ["Multiple-response count", "Ranking / ordered preference"],
        horizontal=True,
        key=f"{name}_answer_pattern_type"
    )

    selected_column = st.selectbox(
        "Choose a question",
        df.columns,
        key=f"{name}_answer_pattern_column"
    )

    separator = st.text_input(
        "Separator between answers",
        value=";",
        key=f"{name}_answer_pattern_separator"
    )

    if not separator:
        st.warning("Enter the separator used between answers, for example ;")
        return

    responses = df[selected_column].dropna().astype(str)

    parsed_rows = []

    for response in responses:
        clean_parts = split_cell_values(response, separator)

        if clean_parts:
            parsed_rows.append(clean_parts)

    if len(parsed_rows) == 0:
        st.warning("No answers found after splitting this column.")
        return

    if analysis_type == "Multiple-response count":
        all_answers = []

        for row in parsed_rows:
            for answer in row:
                all_answers.append(answer)

        counts = pd.Series(all_answers).value_counts().reset_index()
        counts.columns = ["Answer Option", "Selection Count"]

        counts["Percentage of Selections"] = (
            counts["Selection Count"] / counts["Selection Count"].sum() * 100
        ).round(1)

        counts["Percentage of Respondents"] = (
            counts["Selection Count"] / len(parsed_rows) * 100
        ).round(1)

        counts["Label"] = (
            counts["Selection Count"].astype(str)
            + " ("
            + counts["Percentage of Selections"].astype(str)
            + "%)"
        )

        st.subheader("Multiple-response summary")
        st.dataframe(counts, width="stretch")

        fig = px.bar(
            counts,
            x="Answer Option",
            y="Selection Count",
            text="Label",
            title=f"Selected options in: {selected_column}"
        )

        fig.update_traces(textposition="outside", cliponaxis=False)

        st.plotly_chart(fig, width="stretch")

        st.write(f"Valid respondents: **{len(parsed_rows)}**")
        st.write(f"Total selections: **{len(all_answers)}**")
        st.write(f"Unique answer options: **{counts.shape[0]}**")

        if st.button("Generate AI multiple-response interpretation", key=f"{name}_multi_response_ai"):
            ai_context = f"""
Dataset: {name}
Question: {selected_column}
Separator used: {separator}
Valid respondents: {len(parsed_rows)}
Total selections: {len(all_answers)}
Unique answer options: {counts.shape[0]}

Results:
{counts.to_string(index=False)}
"""

            st.write(
                ai_generate(
                    f"{name}: Multiple-response analysis",
                    ai_context,
                    "Interpret the most frequently selected options. Explain that respondents could select more than one answer."
                )
            )

    else:
        ranking_records = []

        for respondent_id, row in enumerate(parsed_rows, start=1):
            for position, answer in enumerate(row, start=1):
                ranking_records.append({
                    "Respondent": respondent_id,
                    "Answer Option": answer,
                    "Rank Position": position
                })

        ranking_df = pd.DataFrame(ranking_records)

        rank_summary = (
            ranking_df
            .groupby("Answer Option")
            .agg(
                Times_Selected=("Answer Option", "count"),
                Average_Rank=("Rank Position", "mean"),
                Best_Rank=("Rank Position", "min"),
                Worst_Rank=("Rank Position", "max")
            )
            .reset_index()
        )

        rank_summary["Average_Rank"] = rank_summary["Average_Rank"].round(2)

        rank_summary = rank_summary.sort_values(
            by=["Average_Rank", "Times_Selected"],
            ascending=[True, False]
        )

        st.subheader("Ranking summary")
        st.write("Lower Average Rank means higher importance.")
        st.dataframe(rank_summary, width="stretch")

        position_table = pd.crosstab(
            ranking_df["Answer Option"],
            ranking_df["Rank Position"]
        )

        st.subheader("Rank position table")
        st.dataframe(position_table, width="stretch")

        first_choice = (
            ranking_df[ranking_df["Rank Position"] == 1]["Answer Option"]
            .value_counts()
            .reset_index()
        )

        first_choice.columns = ["Answer Option", "First Choice Count"]

        total_first = first_choice["First Choice Count"].sum()
        first_choice["Percentage"] = (
            first_choice["First Choice Count"] / max(total_first, 1) * 100
        ).round(1)

        first_choice["Label"] = (
            first_choice["First Choice Count"].astype(str)
            + " ("
            + first_choice["Percentage"].astype(str)
            + "%)"
        )

        st.subheader("First-choice summary")
        st.dataframe(first_choice, width="stretch")

        fig_avg = px.bar(
            rank_summary,
            x="Answer Option",
            y="Average_Rank",
            text="Average_Rank",
            title=f"Average ranking position: {selected_column}"
        )

        fig_avg.update_traces(textposition="outside", cliponaxis=False)

        st.plotly_chart(fig_avg, width="stretch")

        fig_first = px.bar(
            first_choice,
            x="Answer Option",
            y="First Choice Count",
            text="Label",
            title=f"Most common first-choice options: {selected_column}"
        )

        fig_first.update_traces(textposition="outside", cliponaxis=False)

        st.plotly_chart(fig_first, width="stretch")

        full_orders = []

        for row in parsed_rows:
            full_orders.append(" > ".join(row))

        order_summary = pd.Series(full_orders).value_counts().reset_index()
        order_summary.columns = ["Full Ranking Order", "Count"]

        st.subheader("Most repeated full ranking orders")
        st.dataframe(order_summary, width="stretch")

        st.write(f"Valid respondents: **{len(parsed_rows)}**")
        st.write(f"Unique full ranking orders: **{order_summary.shape[0]}**")

        if st.button("Generate AI ranking interpretation", key=f"{name}_ranking_ai"):
            ai_context = f"""
Dataset: {name}
Question: {selected_column}
Separator used: {separator}
Valid respondents: {len(parsed_rows)}

Ranking summary:
{rank_summary.to_string(index=False)}

Rank position table:
{position_table.to_string()}

First-choice summary:
{first_choice.to_string(index=False)}

Most repeated full ranking orders:
{order_summary.head(10).to_string(index=False)}
"""

            st.write(
                ai_generate(
                    f"{name}: Ranking preference analysis",
                    ai_context,
                    "Interpret the ranking order. Explain that lower average rank indicates higher perceived importance. Mention first-choice patterns and repeated ranking orders."
                )
            )


def crosstab_lab(df, name):
    st.markdown(f"### {name} Relationship Lab")

    safe_options = [
        column for column in df.columns
        if column.lower() not in ["id", "id_2", "unnamed"]
    ]

    if len(safe_options) < 2:
        safe_options = list(df.columns)

    c1, c2 = st.columns(2)

    with c1:
        row = st.selectbox(
            "Rows",
            safe_options,
            key=f"{name}_cross_row"
        )

    with c2:
        col = st.selectbox(
            "Columns",
            safe_options,
            key=f"{name}_cross_col"
        )

    if row == col:
        st.warning("Please select two different variables for the relationship analysis.")
        return

    split_answers = st.checkbox(
        "Split multiple answers inside selected variables",
        value=True,
        key=f"{name}_cross_split"
    )

    separator = st.text_input(
        "Separator used between answers",
        value=";",
        key=f"{name}_cross_separator"
    )

    relationship_rows = []

    for _, record in df[[row, col]].dropna(how="all").iterrows():
        row_value_raw = record[row]
        col_value_raw = record[col]

        if pd.isna(row_value_raw) or pd.isna(col_value_raw):
            continue

        if split_answers:
            row_values = split_cell_values(row_value_raw, separator)
            col_values = split_cell_values(col_value_raw, separator)
        else:
            row_values = [str(row_value_raw).strip()]
            col_values = [str(col_value_raw).strip()]

        for row_value in row_values:
            for col_value in col_values:
                relationship_rows.append({
                    "Row Category": row_value,
                    "Column Category": col_value
                })

    relationship_df = pd.DataFrame(relationship_rows)

    if relationship_df.empty:
        st.warning("No valid relationship data found.")
        return

    table = pd.crosstab(
        relationship_df["Row Category"],
        relationship_df["Column Category"]
    )

    st.subheader("Crosstab table")
    st.dataframe(table, width="stretch")

    long_table = (
        table
        .reset_index()
        .melt(
            id_vars="Row Category",
            var_name="Column Category",
            value_name="Count"
        )
    )

    long_table = long_table[long_table["Count"] > 0]

    total_count = long_table["Count"].sum()

    long_table["Percentage"] = (
        long_table["Count"] / max(total_count, 1) * 100
    ).round(1)

    long_table["Label"] = (
        long_table["Count"].astype(str)
        + " ("
        + long_table["Percentage"].astype(str)
        + "%)"
    )

    st.subheader("Chart data with percentages")
    st.dataframe(
        long_table[["Row Category", "Column Category", "Count", "Percentage"]],
        width="stretch"
    )

    fig = px.bar(
        long_table,
        x="Row Category",
        y="Count",
        color="Column Category",
        barmode="group",
        text="Label",
        title=f"{row} vs {col}"
    )

    fig.update_traces(textposition="outside", cliponaxis=False)

    fig.update_layout(
        xaxis_title="Row Category",
        yaxis_title="Count",
        uniformtext_minsize=9,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, width="stretch")

    heatmap_fig = create_heatmap_with_labels(
        table,
        f"Heatmap: {row} vs {col}"
    )

    st.plotly_chart(heatmap_fig, width="stretch")

    if st.button("Generate AI relationship interpretation", key=f"{name}_cross_ai"):
        st.write(
            ai_generate(
                f"{name}: Crosstab Analysis",
                long_table[["Row Category", "Column Category", "Count", "Percentage"]].to_string(index=False),
                f"Rows: {row}; Columns: {col}; Sample size: {len(df)}"
            )
        )


def reliability_lab(df, name):
    st.markdown(f"### {name} Reliability Lab")

    nums = numeric_columns(df)
    default = nums[:8]

    selected = st.multiselect(
        "Select Likert-scale items",
        nums,
        default=default,
        key=f"{name}_alpha_cols"
    )

    if len(selected) < 2:
        st.info("Select at least two numeric items.")
        return

    alpha, n, k = cronbach_alpha(df[selected])

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Cronbach's Alpha",
            "N/A" if alpha is None else round(alpha, 3),
            "Internal consistency"
        )

    with c2:
        metric_card(
            "Reliability",
            alpha_label(alpha),
            "Interpretation"
        )

    with c3:
        metric_card(
            "Valid responses",
            n,
            f"Items selected: {k}"
        )

    corr = df[selected].apply(pd.to_numeric, errors="coerce").corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Inter-item Correlation Heatmap"
    )

    st.plotly_chart(fig, width="stretch")

    if st.button("Generate AI reliability interpretation", key=f"{name}_alpha_ai"):
        result = f"""
Alpha: {alpha}
Interpretation: {alpha_label(alpha)}
Valid responses: {n}
Items: {selected}
"""
        st.write(
            ai_generate(
                f"{name}: Reliability Analysis",
                result,
                "Interpret Cronbach's Alpha carefully and avoid overclaiming."
            )
        )


def text_lab(df, name):
    st.markdown(f"### {name} Text Response Lab")

    candidates = likely_text_columns(df)

    if not candidates:
        st.info("No strong text-response columns detected. You can still choose from all columns.")
        candidates = list(df.columns)

    col = st.selectbox(
        "Choose text/open-ended question",
        candidates,
        key=f"{name}_text_col"
    )

    s = df[col].dropna().astype(str)

    st.write(f"Text responses: {len(s)}")

    st.dataframe(
        pd.DataFrame({"Response": s.head(50)}),
        width="stretch"
    )

    keywords = text_keywords(s)

    st.dataframe(keywords, width="stretch")

    if not keywords.empty:
        keywords["Label"] = keywords["Count"].astype(str)

        fig = px.bar(
            keywords,
            x="Keyword",
            y="Count",
            text="Label",
            title="Top Keywords"
        )

        fig.update_traces(textposition="outside", cliponaxis=False)

        st.plotly_chart(fig, width="stretch")

    if st.button("Generate AI text-theme analysis", key=f"{name}_text_ai"):
        sample = "\n".join(s.head(30).tolist())

        st.write(
            ai_generate(
                f"{name}: Text Response Analysis",
                sample,
                "Identify themes only from the provided responses."
            )
        )


# =========================================================
# Export
# =========================================================

def safe_sheet_name(name):
    invalid_chars = ["[", "]", ":", "*", "?", "/", "\\"]

    name = str(name)

    for char in invalid_chars:
        name = name.replace(char, "_")

    name = name.strip()

    if not name:
        name = "Sheet"

    return name[:31]


def export_report(datasets):
    output = io.BytesIO()
    used_sheet_names = set()

    def unique_sheet_name(raw_name):
        base_name = safe_sheet_name(raw_name)
        sheet_name = base_name
        counter = 1

        while sheet_name in used_sheet_names:
            suffix = f"_{counter}"
            sheet_name = safe_sheet_name(base_name[:31 - len(suffix)] + suffix)
            counter += 1

        used_sheet_names.add(sheet_name)
        return sheet_name

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in datasets.items():
            if df is None:
                continue

            df.head(1000).to_excel(
                writer,
                sheet_name=unique_sheet_name(f"{name}_Raw"),
                index=False
            )

            overview = pd.DataFrame({
                "Metric": [
                    "Rows",
                    "Columns",
                    "Missing Cells",
                    "Numeric Columns",
                    "Text-like Columns"
                ],
                "Value": [
                    df.shape[0],
                    df.shape[1],
                    int(df.isna().sum().sum()),
                    len(numeric_columns(df)),
                    len(likely_text_columns(df))
                ],
            })

            overview.to_excel(
                writer,
                sheet_name=unique_sheet_name(f"{name}_Overview"),
                index=False
            )

            for i, col in enumerate(df.columns[:20], start=1):
                question_summary = summary_table(df, col)

                question_summary.to_excel(
                    writer,
                    sheet_name=unique_sheet_name(f"{name}_Q{i}"),
                    index=False
                )

    return output.getvalue()


# =========================================================
# Main App
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Pro AI Survey Analytics Dashboard</div>
        <div class="hero-sub">
            Upload lecturer and student questionnaire files, explore patterns,
            test reliability, review open-ended responses, compare datasets,
            and generate academic AI interpretations.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.title("Control Panel")
    st.caption("Upload files and configure AI.")

    lecturer_file = st.file_uploader(
        "Lecturer questionnaire",
        type=["xlsx"],
        key="lecturer_upload"
    )

    student_file = st.file_uploader(
        "Student questionnaire",
        type=["xlsx"],
        key="student_upload"
    )

    st.divider()

    st.subheader("AI Settings")

    st.session_state["manual_api_key"] = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("manual_api_key", "")
    )

    st.session_state["selected_model"] = st.text_input(
        "OpenAI model",
        value=st.session_state.get("selected_model", "gpt-5.5")
    )

    st.caption("Each user can enter their own API key. Do not hard-code API keys inside the app.")


try:
    lecturer_df = clean_dataframe(pd.read_excel(lecturer_file)) if lecturer_file else None
except Exception as error:
    lecturer_df = None
    st.error(f"Could not read lecturer file: {error}")

try:
    student_df = clean_dataframe(pd.read_excel(student_file)) if student_file else None
except Exception as error:
    student_df = None
    st.error(f"Could not read student file: {error}")


tabs = st.tabs([
    "🏠 Overview",
    "👩‍🏫 Lecturers",
    "🎓 Students",
    "⚖️ Comparison",
    "📤 Export"
])


with tabs[0]:
    st.markdown("## Project Overview")

    if lecturer_df is None and student_df is None:
        st.info("Upload at least one Excel file from the sidebar to begin.")

    if lecturer_df is not None:
        dataset_overview(lecturer_df, "Lecturers")

    if student_df is not None:
        dataset_overview(student_df, "Students")


with tabs[1]:
    if lecturer_df is None:
        st.info("Upload the lecturer questionnaire file from the sidebar.")
    else:
        sub_tabs = st.tabs([
            "Overview",
            "Quality",
            "Questions",
            "Multi Chart",
            "Filters",
            "Answer Pattern",
            "Relationships",
            "Reliability",
            "Text"
        ])

        with sub_tabs[0]:
            dataset_overview(lecturer_df, "Lecturers")

        with sub_tabs[1]:
            data_quality(lecturer_df, "Lecturers")

        with sub_tabs[2]:
            question_lab(lecturer_df, "Lecturers")

        with sub_tabs[3]:
            multi_variable_chart_lab(lecturer_df, "Lecturers")

        with sub_tabs[4]:
            filter_lab(lecturer_df, "Lecturers")

        with sub_tabs[5]:
            yes_no_pattern_lab(lecturer_df, "Lecturers")

        with sub_tabs[6]:
            crosstab_lab(lecturer_df, "Lecturers")

        with sub_tabs[7]:
            reliability_lab(lecturer_df, "Lecturers")

        with sub_tabs[8]:
            text_lab(lecturer_df, "Lecturers")


with tabs[2]:
    if student_df is None:
        st.info("Upload the student questionnaire file from the sidebar.")
    else:
        sub_tabs = st.tabs([
            "Overview",
            "Quality",
            "Questions",
            "Multi Chart",
            "Filters",
            "Answer Pattern",
            "Relationships",
            "Reliability",
            "Text"
        ])

        with sub_tabs[0]:
            dataset_overview(student_df, "Students")

        with sub_tabs[1]:
            data_quality(student_df, "Students")

        with sub_tabs[2]:
            question_lab(student_df, "Students")

        with sub_tabs[3]:
            multi_variable_chart_lab(student_df, "Students")

        with sub_tabs[4]:
            filter_lab(student_df, "Students")

        with sub_tabs[5]:
            yes_no_pattern_lab(student_df, "Students")

        with sub_tabs[6]:
            crosstab_lab(student_df, "Students")

        with sub_tabs[7]:
            reliability_lab(student_df, "Students")

        with sub_tabs[8]:
            text_lab(student_df, "Students")


with tabs[3]:
    st.markdown("## Lecturer vs Student Comparison")

    if lecturer_df is None or student_df is None:
        st.info("Upload both files to enable comparison.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            lcol = st.selectbox(
                "Lecturer question",
                lecturer_df.columns,
                key="cmp_lcol"
            )

            split_l = st.checkbox(
                "Split lecturer answers",
                value=True,
                key="cmp_l_split"
            )

            if split_l:
                lsum = split_summary_table(lecturer_df, lcol, ";")
            else:
                lsum = summary_table(lecturer_df, lcol)

            st.dataframe(lsum[["Response", "Count", "Percentage"]], width="stretch")

            lfig = px.bar(
                lsum,
                x="Response",
                y="Count",
                text="Label",
                title="Lecturer responses"
            )

            lfig.update_traces(textposition="outside", cliponaxis=False)

            st.plotly_chart(lfig, width="stretch")

        with c2:
            scol = st.selectbox(
                "Student question",
                student_df.columns,
                key="cmp_scol"
            )

            split_s = st.checkbox(
                "Split student answers",
                value=True,
                key="cmp_s_split"
            )

            if split_s:
                ssum = split_summary_table(student_df, scol, ";")
            else:
                ssum = summary_table(student_df, scol)

            st.dataframe(ssum[["Response", "Count", "Percentage"]], width="stretch")

            sfig = px.bar(
                ssum,
                x="Response",
                y="Count",
                text="Label",
                title="Student responses"
            )

            sfig.update_traces(textposition="outside", cliponaxis=False)

            st.plotly_chart(sfig, width="stretch")

        if st.button("Generate AI comparative analysis"):
            results = f"""
Lecturer question:
{lcol}

Lecturer results:
{lsum[["Response", "Count", "Percentage"]].to_string(index=False)}

Student question:
{scol}

Student results:
{ssum[["Response", "Count", "Percentage"]].to_string(index=False)}
"""

            st.write(
                ai_generate(
                    "Lecturer vs Student Comparative Analysis",
                    results,
                    "Do not assume both questions measure the same construct unless clearly indicated."
                )
            )


with tabs[4]:
    st.markdown("## Export Analysis")

    st.write(
        "Download an Excel report containing raw previews, overview metrics, "
        "and summaries for the first 20 columns of each uploaded dataset."
    )

    if lecturer_df is None and student_df is None:
        st.info("Upload data first.")
    else:
        report = export_report({
            "Lecturers": lecturer_df,
            "Students": student_df
        })

        st.download_button(
            "Download Excel Report",
            data=report,
            file_name="pro_survey_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
