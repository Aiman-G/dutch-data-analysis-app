from embedding_atlas.streamlit import embedding_atlas
#from embedding_atlas.projection import compute_text_projection
from utils.data_loader import load_data_noun_MiniLM_L12_v2_embeddings



df = load_data_noun_MiniLM_L12_v2_embeddings()

# # Compute text embedding and projection of the embedding
# compute_text_projection(
#     data_frame=df,
#     text="lemma",x="projection_x", y="projection_y"
# )

# Create an Embedding Atlas component for a given data frame
value = embedding_atlas(
    df, text="lemma",
    x="projection_x", y="projection_y",  labels="automatic",
    show_table=True, show_embedding=True, show_charts=True
)

