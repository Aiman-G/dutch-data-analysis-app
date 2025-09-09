import pandas as pd
import streamlit as st

@st.cache_data
def load_data_noun_embeddings():
    return pd.read_csv("data/mc4_nl_cleaned_micro_embedding_UMAP.csv")

@st.cache_data
def load_data_noun_MiniLM_L12_v2_embeddings():
    return pd.read_csv("data/mc4_nl_cleaned_large_embedding_UMAP_MiniLM-L12-v2.csv")
