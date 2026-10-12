from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# App configuration and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Marketing Response Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
      div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
      }
      .small-note {font-size: 0.88rem; color: #777;}
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "marketing_model_bundle.joblib"


@st.cache_resource(show_spinner="Loading the trained LightGBM model...")
def load_bundle(path: str):
    # Only load a model bundle created by your own notebook/project.
    return joblib.load(path)


def clean_target(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def engineer_features_for_app(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the dashboard transformation aligned with engineer_features() in the notebook."""
    df = df.copy()
    if "pdays" in df.columns:
        df["previously_contacted"] = (df["pdays"] != 999).astype(int)
        df["pdays"] = df["pdays"].replace(999, np.nan)
    if "pmonths" in df.columns:
        df["pmonths"] = df["pmonths"].replace(999, np.nan)
    return df


def prepare_for_model(raw_row: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply feature engineering and the saved LightGBM category levels to one row."""
    row = engineer_features_for_app(raw_row)

    category_levels = bundle["category_levels"]
    categorical_cols = bundle["categorical_cols"]

    for col in categorical_cols:
        if col in row.columns:
            values = row[col].astype("string").fillna("Missing")
            row[col] = pd.Categorical(
                values,
                categories=category_levels[col],
            )

    return row.reindex(columns=bundle["feature_columns"])


def get_numeric_input_config(training_data: pd.DataFrame, col: str):
    """Derive sensible form limits/defaults from training data, ignoring 999 sentinels."""
    values = pd.to_numeric(training_data[col], errors="coerce")
    if col in {"pdays", "pmonths"}:
        values = values[values != 999]
    values = values.dropna()

    if values.empty:
        return 0.0, 100.0, 0.0, 1.0

    min_value = float(values.min())
    max_value = float(values.max())
    default_value = float(values.median())

    integer_columns = {"custAge", "campaign", "previous", "pastEmail"}
    if col in integer_columns:
        min_value = float(np.floor(min_value))
        max_value = float(np.ceil(max_value))
        default_value = float(round(default_value))
        step = 1.0
    else:
        step = 0.1

    if min_value == max_value:
        min_value -= step
        max_value += step

    default_value = min(max(default_value, min_value), max_value)
    return min_value, max_value, default_value, step


if not BUNDLE_PATH.exists():
    st.title("Marketing Campaign Response Dashboard")
    st.warning("The model bundle has not been created yet.")
    st.markdown(
        "Run the final LightGBM training cell in your notebook, then run the "
        "**Export dashboard bundle** cell included with this dashboard. It creates:"
    )
    st.code(str(BUNDLE_PATH), language="text")
    st.markdown(
        "Keep `streamlit_app.py` and `marketing_model_bundle.joblib` in the same folder, "
        "then rerun this app. See `README_streamlit.md` for setup instructions."
    )
    st.stop()

try:
    bundle = load_bundle(str(BUNDLE_PATH))
except Exception as exc:
    st.error(f"Unable to load the model bundle: {exc}")
    st.info("Recreate the bundle from the notebook and make sure it matches this app.")
    st.stop()

required_keys = {
    "model",
    "input_feature_columns",
    "feature_columns",
    "categorical_cols",
    "category_levels",
    "training_data",
}
missing_keys = required_keys.difference(bundle)
if missing_keys:
    st.error(f"The saved model bundle is missing required fields: {sorted(missing_keys)}")
    st.stop()

model = bundle["model"]
training_data = bundle["training_data"].copy()
input_feature_columns = bundle["input_feature_columns"]
categorical_cols = bundle["categorical_cols"]

if "responded" not in training_data.columns:
    st.error("The dashboard training data does not contain the 'responded' target column.")
    st.stop()

# Ensure overview calculations only use rows with a known target.
target_clean = clean_target(training_data["responded"])
training_data = training_data.loc[target_clean.isin(["yes", "no"])].copy()
training_data["responded_clean"] = clean_target(training_data["responded"])
training_data["response_flag"] = (training_data["responded_clean"] == "yes").astype(int)

st.title("Marketing Campaign Response Dashboard")
st.caption("Insurance campaign analytics · Customer response prediction · Tuned LightGBM")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a view",
        ["Campaign Overview", "Predict a Customer", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("The model score is an estimate, not a guarantee of customer response.")


# -----------------------------------------------------------------------------
# Page 1: Campaign overview
# -----------------------------------------------------------------------------
if page == "Campaign Overview":
    st.subheader("Campaign Overview")
    total_customers = len(training_data)
    total_responders = int(training_data["response_flag"].sum())
    response_rate = (total_responders / total_customers * 100) if total_customers else 0.0
    non_responders = total_customers - total_responders

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers in training data", f"{total_customers:,}")
    k2.metric("Responders", f"{total_responders:,}")
    k3.metric("Non-responders", f"{non_responders:,}")
    k4.metric("Observed response rate", f"{response_rate:.2f}%")

    left, right = st.columns(2)
    with left:
        counts = (
            training_data["responded_clean"]
            .value_counts()
            .reindex(["no", "yes"], fill_value=0)
            .rename_axis("Response")
            .reset_index(name="Customers")
        )
        counts["Response"] = counts["Response"].map({"no": "No response", "yes": "Responded"})
        fig = px.bar(
            counts,
            x="Response",
            y="Customers",
            text="Customers",
            title="Observed response distribution",
            color="Response",
            color_discrete_map={"No response": "#94A3B8", "Responded": "#2563EB"},
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        category_candidates = [
            col for col in ["profession", "month", "contact", "poutcome", "schooling"]
            if col in training_data.columns
        ]
        if category_candidates:
            chosen_category = st.selectbox("Explore response rate by", category_candidates)
            plot_data = training_data[[chosen_category, "response_flag"]].copy()
            plot_data[chosen_category] = plot_data[chosen_category].fillna("Missing").astype(str)
            response_rates = (
                plot_data.groupby(chosen_category, dropna=False)
                .agg(response_rate=("response_flag", "mean"), customers=("response_flag", "size"))
                .reset_index()
            )
            response_rates["response_rate_pct"] = response_rates["response_rate"] * 100
            response_rates = response_rates.sort_values("response_rate_pct", ascending=False)
            fig = px.bar(
                response_rates,
                x=chosen_category,
                y="response_rate_pct",
                hover_data={"customers": True, "response_rate_pct": ":.2f"},
                title=f"Response rate by {chosen_category}",
                labels={chosen_category: chosen_category.replace("_", " ").title(), "response_rate_pct": "Response rate (%)"},
                template="plotly_white",
                color="response_rate_pct",
                color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Response rate (%)", coloraxis_showscale=False)
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview training data"):
        visible_cols = [c for c in training_data.columns if c not in {"response_flag", "responded_clean"}]
        st.dataframe(training_data[visible_cols].head(100), use_container_width=True)

    st.info(
        "The response rate and charts describe the labelled training dataset. "
        "They are descriptive summaries, not estimates of performance on future campaigns."
    )


# -----------------------------------------------------------------------------
# Page 2: Individual customer prediction
# -----------------------------------------------------------------------------
elif page == "Predict a Customer":
    st.subheader("Predict a Customer's Campaign Response")
    st.write("Enter customer and campaign details. The selected LightGBM model will return a response score and predicted class.")

    category_fields = [c for c in input_feature_columns if c in categorical_cols]
    numeric_fields = [
        c for c in input_feature_columns
        if c not in categorical_cols and c not in {"pdays", "pmonths"}
    ]

    with st.form("customer_prediction_form"):
        cat_col, num_col = st.columns(2, gap="large")
        values = {}

        with cat_col:
            st.markdown("#### Customer and campaign categories")
            for col in category_fields:
                options = [str(v) for v in bundle["category_levels"].get(col, [])]
                if not options:
                    options = ["Missing"]
                preferred_defaults = {
                    "profession": "admin.",
                    "marital": "married",
                    "schooling": "university.degree",
                    "default": "no",
                    "housing": "no",
                    "loan": "no",
                    "contact": "cellular",
                    "month": "may",
                    "day_of_week": "mon",
                    "poutcome": "nonexistent",
                }
                preferred = preferred_defaults.get(col)
                default_index = options.index(preferred) if preferred in options else 0
                values[col] = st.selectbox(
                    col.replace("_", " ").replace(".", " ").title(),
                    options=options,
                    index=default_index,
                    key=f"input_{col}",
                )

        with num_col:
            st.markdown("#### Numeric attributes")
            for col in numeric_fields:
                minimum, maximum, default, step = get_numeric_input_config(training_data, col)
                values[col] = st.number_input(
                    col.replace("_", " ").replace(".", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    key=f"input_{col}",
                )

            st.markdown("#### Previous campaign contact")
            previously_contacted_input = st.checkbox(
                "Customer was contacted in a previous campaign",
                value=False,
            )
            if previously_contacted_input:
                pdays_input = st.number_input(
                    "Days since previous contact (pdays)",
                    min_value=0,
                    max_value=998,
                    value=30,
                    step=1,
                )
                pmonths_input = st.number_input(
                    "Months since previous contact (pmonths)",
                    min_value=0.0,
                    max_value=60.0,
                    value=1.0,
                    step=0.1,
                )
                values["pdays"] = pdays_input
                values["pmonths"] = pmonths_input
            else:
                # In this dataset, 999 denotes no previous contact.
                values["pdays"] = 999
                values["pmonths"] = 999

        submit_prediction = st.form_submit_button(
            "Predict response",
            type="primary",
            use_container_width=True,
        )

    if submit_prediction:
        missing_input_cols = set(input_feature_columns) - set(values)
        if missing_input_cols:
            st.error(f"The form is missing these model inputs: {sorted(missing_input_cols)}")
        else:
            raw_row = pd.DataFrame(
                [{col: values[col] for col in input_feature_columns}],
                columns=input_feature_columns,
            )
            model_row = prepare_for_model(raw_row, bundle)

            try:
                response_score = float(model.predict_proba(model_row)[0, 1])
                predicted_response = "yes" if response_score >= 0.5 else "no"

                st.divider()
                result_col, score_col = st.columns([1, 2])
                with result_col:
                    if predicted_response == "yes":
                        st.success("Predicted response: YES")
                    else:
                        st.warning("Predicted response: NO")
                    st.metric("Predicted class", predicted_response.upper())
                with score_col:
                    st.metric("Estimated response score", f"{response_score:.1%}")
                    st.progress(response_score)
                    st.caption(
                        "The class uses a 0.50 threshold. Because the model was trained with class-imbalance weighting, "
                        "the score may not be perfectly calibrated and should not be interpreted as a guarantee."
                    )

                with st.expander("View the submitted customer details"):
                    st.dataframe(raw_row, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check that the saved model bundle was exported after final training and uses the same feature columns.")


# -----------------------------------------------------------------------------
# Page 3: Model performance
# -----------------------------------------------------------------------------
else:
    st.subheader("Model Performance")
    st.write("Performance figures below come from the held-out validation set, not the unseen test dataset.")

    metric_rows = bundle.get("validation_metrics", [])
    metrics_df = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()

    if not metrics_df.empty and "Model" in metrics_df.columns:
        main_row = metrics_df[metrics_df["Model"].astype(str).str.contains("Tuned LightGBM", case=False)]
        if main_row.empty:
            main_row = metrics_df[metrics_df["Model"].astype(str).str.contains("LightGBM", case=False)]
        if not main_row.empty:
            row = main_row.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            for container, label, key in [
                (c1, "Accuracy", "Accuracy"),
                (c2, "Precision", "Precision"),
                (c3, "Recall", "Recall"),
                (c4, "F1-score", "F1-score"),
            ]:
                value = row.get(key, np.nan)
                container.metric(label, f"{float(value):.3f}" if pd.notna(value) else "—")

        display_cols = [
            col for col in ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]
            if col in metrics_df.columns
        ]
        st.markdown("#### Validation comparison")
        st.dataframe(metrics_df[display_cols].round(4), hide_index=True, use_container_width=True)

        chart_metric = st.selectbox(
            "Compare models by metric",
            [m for m in ["F1-score", "Precision", "Recall", "ROC-AUC", "Average Precision", "Accuracy"] if m in metrics_df.columns],
            index=0,
        )
        fig = px.bar(
            metrics_df,
            x="Model",
            y=chart_metric,
            color="Model",
            text=chart_metric,
            title=f"Validation {chart_metric}",
            template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], showlegend=False, xaxis_title="", yaxis_title=chart_metric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No validation metrics were saved in the model bundle. Re-run the bundle-export cell after model evaluation.")

    cm = bundle.get("validation_confusion_matrix")
    if cm is not None:
        st.markdown("#### Tuned LightGBM — validation confusion matrix")
        cm_array = np.asarray(cm)
        fig = px.imshow(
            cm_array,
            x=["Predicted no response", "Predicted responded"],
            y=["Actual no response", "Actual responded"],
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            labels={"x": "Predicted label", "y": "Actual label", "color": "Customers"},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    importances = getattr(model, "feature_importances_", None)
    if importances is not None and len(importances) == len(bundle["feature_columns"]):
        st.markdown("#### LightGBM feature importance")
        importance_df = pd.DataFrame({
            "Feature": bundle["feature_columns"],
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(15).sort_values("Importance")
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 features by split importance",
            template="plotly_white",
        )
        fig.update_layout(yaxis_title="", xaxis_title="Split importance")
        st.plotly_chart(fig, use_container_width=True)

    best_params = bundle.get("best_params")
    if best_params:
        with st.expander("View selected LightGBM hyperparameters"):
            st.json(best_params)

    st.caption(
        "Feature importance describes how the fitted trees used features; it does not establish causation. "
        "Validation metrics estimate performance and are not test-set scores."
    )
