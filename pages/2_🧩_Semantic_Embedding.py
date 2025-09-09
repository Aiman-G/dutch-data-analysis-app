from embedding_atlas.streamlit import embedding_atlas
#from embedding_atlas.projection import compute_text_projection
from utils.data_loader import load_data_noun_embeddings
import streamlit as st


st.info(
    "This visualization shows the **semantic embeddings of nouns** extracted from the data. \n\n"
    "- Each point represents a noun.\n"
    "- 🔍 Zoom in and out to explore details at different scales.\n"
    "- 🎨 Color the points by **article, suffix, prefix**, or other features.\n"
    "- 📰 See which articles dominate within different semantic topics.\n"
    "- ◻️ Use the **Select (dashed square) tool** to highlight points and view details about them in the bar plots or additional charts.\n\n"
    ,icon="ℹ️"
)



df = load_data_noun_embeddings()

# # Compute text embedding and projection of the embedding
# compute_text_projection(
#     data_frame=df,
#     text="lemma",x="projection_x", y="projection_y", neighbors="neighbors"
# )

# Create the Embedding Atlas visualization
value = embedding_atlas(
    df,
    text="lemma",
    x="projection_x",
    y="projection_y",
    neighbors="neighbors",
    labels="automatic",
    show_table=True,
    show_embedding=True,
    show_charts=True
)

