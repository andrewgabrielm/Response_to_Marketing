import streamlit as st
import pandas as pd
#Creates charts with relatively little code.
import plotly.express as px
#Creates the confusion matrix visualisations with more control.
import plotly.graph_objects as go

st.set_page_config(
    page_title = "marketing Campaign",
    layout = "wide"
)

st.title("Marketing Campaign Analysis")
st.caption("Insurance campaign response analysis")


# Add file to the uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload training dataset",
    type = ["xlsx","csv"]
)

# Read the uploaded file
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset loaded succesfully")
    # Display a preview 
    st.subheader("Dataset Preview")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head())
else: 
    st.info("Upload your training dataset in the sidebar")

# calculate and display KPIs
# 4 summary figures , Total customers, responder, non responders, overall response rate

# Step 1: cleaning target column
df = df.dropna(subset = ["responded"]).copy()

df["responded"] = (
    df["responded"].astype(str).str.strip().str.lower()
)

df = df[df["responded"].isin(["yes","no"])].copy()

# Step 2: Calculate the KPIs
total_customers = len(df)
responders = (df["responded"] == "yes").sum()
non_responders = (df["responded"] == "no").sum()

response_rate = (
    responders / total_customers
    if total_customers > 0
    else 0
)

# Step 3 Display the figures in Streamlit
st.subheader("Campaign Sumary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Responders", f"{responders:,}")
col3.metric("Non-responders", f"{non_responders:,}")
col4.metric("Response Rate", f"{response_rate:.1%}")

# Step 4 Add the campaign charts
# 1. Response distribution
# 2. Responsse rate by attribute 

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Response Distribution")

    distribution = (
        df["responded"]
        .value_counts()
        .rename_axis("Response")
        .reset_index(name="Customers")
    )

    fig = px.bar(
        distribution,
        x="Response",
        y="Customers",
        color="Response",
        color_discrete_map={
            "yes": "#2f6b4f",
            "no": "#d9e5dd"
        },
        template="simple_white"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

# Step 5 Modelevalution Tab
overview_tab, evaluation_tab = st.tabs([
    "Campaign Overview",
    "Model Evaluation"
])
with overview_tab:
    confusion_matrices = {
    "Tuned LightGBM": [
        [1280, 182],
        [67, 119]
    ],
    "Tuned Logistic Regression": [
        [1256, 206],
        [63, 123]
    ]
    }

def calculate_metrics(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    total = tn + fp + fn + tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    accuracy = (tn + tp) / total if total else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

# Calculate the evaluation metrics
validation_results = pd.DataFrame.from_dict(
    {
        name: calculate_metrics(matrix)
        for name, matrix in confusion_matrices.items()
    },
    orient="index"
)

validation_results.index.name = "Model"

# Display the metrics 
with evaluation_tab:
    st.subheader("Held-out Validation Results")

    st.caption(
        "Performance on the validation split. "
        "These are not test-set scores."
    )

    st.dataframe(
        validation_results.style.format("{:.3f}"),
        use_container_width=True
    )

    selected_metric = st.selectbox(
        "Compare metric",
        ["Accuracy", "Precision", "Recall", "F1-score"]
    )

    chart_data = (
        validation_results[[selected_metric]]
        .reset_index()
    )

    fig = px.bar(
        chart_data,
        x="Model",
        y=selected_metric,
        color_discrete_sequence=["#2f6b4f"],
        template="simple_white",
        range_y=[0, 1],
        title=f"{selected_metric} Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

# Visualises the confusion metrics

st.subheader("Confusion Matrices")

cols = st.columns(2)

for col, (model_name, matrix) in zip(cols,confusion_matrices.items()):
    with col:
        st.markdown(f"**{model_name}**")

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["No response", "Responded"],
                y=["No response", "Responded"],
                colorscale=[
                    [0, "#f4f8f5"],
                    [0.5, "#b8d1c0"],
                    [1, "#2f6b4f"]
                ],
                showscale=False,
                text=matrix,
                texttemplate="%{text}"
                )
            )

        fig.update_layout(
                template="simple_white",
                height=320,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                yaxis=dict(autorange="reversed")
        )

        st.plotly_chart(
                fig,
                use_container_width=True
        )

# ROC-AUC and averaage precision

validation_results = pd.read_csv(
    "data/processed/validation_metrics.csv",
    index_col=0
)


