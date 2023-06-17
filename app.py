# from operator import index
# import plotly.express as px
import streamlit as st
import pandas as pd
import os

import pandas_profiling as pdp
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model, load_model

df = ""

with st.sidebar:
    # st.image()
    st.title("AutoStreamML")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info(
        "Thid application allows you to build an automated ML pipeline using Streamlit")

if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your data for modelling!")
    if data := st.file_uploader("Upload your dataset here"):
        df = pd.read_csv(data, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning")
    selection = st.selectbox("Select your target attribute: ", df.columns)
    if st.button("Train the model"):
        setup(df, target=selection, silent=True)
        setup_df = pull()
        st.info("This is the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the model", f,
                           file_name="trained_model.pkl")
