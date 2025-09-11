from embedding_atlas.streamlit import embedding_atlas
# from embedding_atlas.projection import compute_text_projection
from utils.data_loader import load_data_noun_embeddings
import streamlit as st

# Create tabs
tab1, tab2 = st.tabs(["🔎 Semantic Embedding Tool", "📖 How to Use"])

# ---------------- TAB 1: The Embedding Tool ----------------
with tab1:
    st.info(
        "This visualization shows the **semantic embeddings of nouns** extracted from the data. \n\n"
        "It will show you how nouns are grouped by their semantic similarity (their meaning).\n\n"
        "You can search for a noun, see which cluster (group) it belongs to, "
        "and check the dominant article (De or Het) in that group. To see the articles of the group:\n\n"
        "- Select the **article column** from the Color dropdown.\n"
        "- Each point represents a noun.\n"
        "- 🔍 Zoom in and out to explore details at different scales.\n"
        "- 🎨 Color the points by **article, suffix, prefix**, or other features.\n"
        "- 📰 See which articles dominate within different semantic topics.\n"
        "- ◻️ Use the **Select (dashed square) tool** to highlight points and view details about them in bar plots or additional charts.\n"
        ,icon="ℹ️"
    )

    # Load data
    df = load_data_noun_embeddings()

    # Embedding Atlas visualization
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

# ---------------- TAB 2: How to Use ----------------
with tab2:
    st.header("📖 How to Use the Tool")
    st.markdown(
        """
        This tool lets you explore the **semantic embeddings of Dutch nouns** and analyze how they group together.

        ### Steps to Use:
        1. **Search for a noun** in the search bar to locate it in the embedding space.  
        2. **Explore the clusters** – nouns that are semantically similar are positioned close together.  
        3. **Color the points** based on:
           - Article (De/Het)  
           - Word suffix  
           - Word prefix  
        4. **Zoom and Pan** to explore details at different scales.  
        5. **Select a cluster** using the **dashed square tool** to see more details in charts and the table.  

        ---
        """
    )

    # Add illustrative figures (replace with your own paths or URLs)
    st.image("assets/search_noun.png", caption="Searching for a noun", use_container_width=True)
    st.image("assets/density_map.png", caption="Coloring nouns by article and choosing density style", use_container_width=True)
    st.image("assets/selection_tool.png", caption="Selecting a cluster to analyze", use_container_width=True)
