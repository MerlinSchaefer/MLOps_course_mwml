from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
from config import config
from tagifai import main, utils

# load dataframe
@st.cache()
def load_data():
    projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
    return pd.read_csv(projects_fp)

df = load_data()
# load atrifacts and metrics of last run
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)


def display_metric_cols(performance: Dict, round_to: int = 2) -> None:
    cols = st.columns(len(performance))
    for col, key in zip(cols, performance.keys()):
        col.metric(f"{key}", round(performance[key], round_to))
    # col1.metric("Overall F1:",round(performance["overall"]["f1"],2))
    # col2.metric("Overall Precision:",round(performance["overall"]["precision"],2))
    # col3.metric("Overall Recall:",round(performance["overall"]["recall"],2))
    # col4.metric("Overall Samples:",round(performance["overall"]["num_samples"],2))


st.title("Tagifai Dashboard")

# Sections
st.header("🔢 Data")
st.write("## Dataset used for training")

st.write(
    "This dataset contains title and description of ML articles, the tags are the topics this article belongs to."
)
st.write(f"Project count: {len(df)}")

st.dataframe(df)

st.header("📊 Performance")
st.write("## Overall performance")
display_metric_cols(performance=performance["overall"])
st.write("## In depth performance")
tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
display_metric_cols(performance=performance["class"][tag])
tag = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
display_metric_cols(performance=performance["slices"][tag])


st.header("🚀 Inference")

text = st.text_input("Enter text:", "Transfer learning with transformers for text classification.")
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
for entry in range(len(prediction)):
    st.write(f"**Your input:** {prediction[entry]['input_text']}")
    st.write(f"**Predicted tag:** {prediction[entry]['predicted_tag']}")