from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.decomposition import PCA

from ultragat.config import ExperimentConfig
from ultragat.metrics import classification_margin, entropy
from ultragat.training import robustness_sweep, train_model

st.set_page_config(page_title="Ultra-GAT Lab", page_icon="U", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem;}
    [data-testid="stMetric"] {border-left: 3px solid #ef5b3f; padding-left: 1rem;}
    h1, h2, h3, p, label {letter-spacing: 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def fit_experiment(config_values: tuple):
    config = ExperimentConfig(**dict(config_values))
    return train_model(config, save_artifact=False)


st.title("Ultra-GAT Lab")
st.caption("Graph attention robustness, uncertainty, and failure geometry in one reproducible lab.")

with st.sidebar:
    st.header("Experiment")
    dataset = st.selectbox("Dataset", ["CiteSeer", "Cora", "PubMed"])
    model_name = st.segmented_control("Model", ["gcn", "gat", "gatv2"], default="gatv2")
    epochs = st.slider("Maximum epochs", 50, 600, 250, 50)
    seed = st.number_input("Seed", min_value=0, max_value=100_000, value=42)
    structural = st.toggle("Structural features", value=True)
    run = st.button("Run experiment", type="primary", use_container_width=True)
    st.divider()
    st.markdown(
        "[Source code](https://github.com/Saroswat/Ultra-GAT-Robust-Graph-Attention-Networks-with-Failure-Geometry-Analysis)"
    )

if run:
    settings = replace(
        ExperimentConfig(),
        dataset=dataset,
        model=model_name,
        epochs=epochs,
        patience=min(80, max(20, epochs // 4)),
        seed=int(seed),
        structural_features=structural,
    )
    st.session_state["experiment"] = fit_experiment(tuple(asdict(settings).items()))

if "experiment" not in st.session_state:
    st.info("Choose an experiment in the sidebar, then run it to unlock the analysis workspace.")
    st.markdown(
        """
        **What the lab measures**

        - Clean node-classification accuracy and expected calibration error
        - Sensitivity to missing edges and corrupted node features
        - Embedding geometry, confidence, margins, and individual failures
        - Reproducible configurations and model artifacts from the command line
        """
    )
    st.stop()

result, model, data = st.session_state["experiment"]
device = data.x.device
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    embeddings = model.embed(data.x, data.edge_index)
probabilities = logits.softmax(dim=-1)
predictions = logits.argmax(dim=-1)
uncertainty = entropy(logits)
margins = classification_margin(logits)

metric_columns = st.columns(4)
metric_columns[0].metric("Test accuracy", f"{result.test_accuracy:.1%}")
metric_columns[1].metric("Validation accuracy", f"{result.validation_accuracy:.1%}")
metric_columns[2].metric("Calibration error", f"{result.ece:.3f}")
metric_columns[3].metric("Parameters", f"{result.parameters:,}")

overview, robustness, failures, run_details = st.tabs(
    ["Embedding map", "Robustness", "Failure geometry", "Run details"]
)

with overview:
    projected = PCA(n_components=2, random_state=result.seed).fit_transform(
        embeddings.detach().cpu().numpy()
    )
    frame = pd.DataFrame(
        {
            "x": projected[:, 0],
            "y": projected[:, 1],
            "true class": data.y.cpu().numpy().astype(str),
            "predicted class": predictions.cpu().numpy().astype(str),
            "confidence": probabilities.max(dim=-1).values.cpu().numpy(),
            "correct": np.where(predictions.cpu() == data.y.cpu(), "correct", "incorrect"),
            "node": np.arange(data.num_nodes),
        }
    )
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="true class",
        symbol="correct",
        hover_data=["node", "predicted class", "confidence"],
        opacity=0.72,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    figure.update_layout(height=610, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)

with robustness:
    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    with st.spinner("Stress-testing edges and features..."):
        edge_result = robustness_sweep(model, data, levels, mode="edges")
        feature_result = robustness_sweep(model, data, levels, mode="features")
    robustness_frame = pd.DataFrame(
        {
            "corruption": levels * 2,
            "accuracy": edge_result["accuracy"] + feature_result["accuracy"],
            "stress test": ["Missing edges"] * len(levels) + ["Masked features"] * len(levels),
        }
    )
    figure = px.line(
        robustness_frame,
        x="corruption",
        y="accuracy",
        color="stress test",
        markers=True,
        color_discrete_map={"Missing edges": "#147d92", "Masked features": "#ef5b3f"},
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    figure.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)
    left, right = st.columns(2)
    left.metric("Edge robustness AUC", f"{edge_result['auc']:.3f}")
    right.metric("Feature robustness AUC", f"{feature_result['auc']:.3f}")

with failures:
    test_nodes = torch.where(data.test_mask)[0]
    failed = test_nodes[predictions[test_nodes] != data.y[test_nodes]]
    failed = failed[torch.argsort(uncertainty[failed], descending=True)]
    selected = st.selectbox(
        "Inspect a misclassified node",
        failed.cpu().tolist(),
        format_func=lambda node: f"Node {node}",
    )
    incoming = data.edge_index[1] == selected
    outgoing = data.edge_index[0] == selected
    neighbors = torch.unique(
        torch.cat([data.edge_index[0, incoming], data.edge_index[1, outgoing]])
    )
    node_columns = st.columns(4)
    node_columns[0].metric("True class", int(data.y[selected]))
    node_columns[1].metric("Predicted class", int(predictions[selected]))
    node_columns[2].metric("Confidence", f"{float(probabilities[selected].max()):.1%}")
    node_columns[3].metric("Graph neighbors", int(neighbors.numel()))
    class_frame = pd.DataFrame(
        {
            "class": np.arange(probabilities.shape[1]).astype(str),
            "probability": probabilities[selected].cpu(),
        }
    )
    st.plotly_chart(
        px.bar(class_frame, x="class", y="probability", color_discrete_sequence=["#ef5b3f"]),
        use_container_width=True,
    )
    st.caption(
        f"Decision margin: {float(margins[selected]):.3f} | "
        f"Entropy: {float(uncertainty[selected]):.3f}"
    )

with run_details:
    st.json(asdict(result))
    st.caption(
        f"Compute device: {device.type}. Best checkpoint selected at epoch {result.best_epoch}."
    )
