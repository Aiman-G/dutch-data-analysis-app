import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import networkx as nx
from utils.data_loader import load_data_noun_embeddings



st.info(
    "⚡ This page contains interactive plots with a large dataset. "
    "Please be patient — plots may take a few moments to update."
)

# Load data
df = load_data_noun_embeddings()



# st.write(len(df))                 # total rows
# st.write(df["lemma"].nunique())   # unique lemmas
# st.write(df["article"].unique())  # all articles


# Sidebar global filters
st.sidebar.header("Filters")
article_filter = st.sidebar.multiselect("Select Article(s)", options=df["article"].unique(), default=df["article"].unique())
length_range = st.sidebar.slider("Word Length Range", int(df["length"].min()), int(df["length"].max()), (2, 15))

filtered_df = df[(df["article"].isin(article_filter)) & (df["length"].between(length_range[0], length_range[1]))]

# Tabs
# overview_tab, umap_tab, morph_tab, length_tab, network_tab, spotlight_tab, density_tab = st.tabs([
#     "📊 Overview", "🌀 UMAP Explorer", "🔤 Morphology", "⌛ Length vs Article", "🧬 Morphological Networks", "📖 Word Spotlight", "🌍 Density View"])

overview_tab, umap_tab, morph_tab, length_tab, spotlight_tab = st.tabs([
    "📊 Overview", "🌀 UMAP Explorer", "🔤 Morphology", "⌛ Length vs Article",  "📖 Word Spotlight"])



# --- Overview ---
with overview_tab:
    st.subheader("Dataset Overview")
    st.metric("Total Words", len(filtered_df))
    st.metric("Unique Suffix-2", filtered_df["suffix_2"].nunique())
    st.metric("Unique Prefix-2", filtered_df["prefix_2"].nunique())

    st.write("### Word Length Distribution")
    bins = st.slider("Number of bins", 5, 50, 20)
    fig = px.histogram(filtered_df, x="length", nbins=bins, color="article", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

# --- UMAP Explorer ---
with umap_tab:
    st.subheader("Embedding Projection (UMAP)")
    color_by = st.selectbox("Color By", ["article", "length", "suffix_2", "prefix_2"])
    point_size = st.slider("Point Size", 3, 15, 6)
    opacity = st.slider("Opacity", 0.1, 1.0, 0.7)

    fig = px.scatter(
        filtered_df,
        x="projection_x", y="projection_y",
        color=color_by,
        size_max=point_size,
        opacity=opacity,
        hover_data=["lemma", "length", "suffix_3", "prefix_3", "article"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Search a Lemma")
    query = st.text_input("Enter lemma")
    if query:
        match = filtered_df[filtered_df["lemma"] == query]
        if not match.empty:
            st.success(f"Found {query}")
            fig_highlight = px.scatter(
                filtered_df, x="projection_x", y="projection_y", opacity=0.3,
                color=color_by, hover_data=["lemma"]
            )
            fig_highlight.add_scatter(
                x=match["projection_x"], y=match["projection_y"],
                mode="markers+text", text=match["lemma"], textposition="top center",
                marker=dict(size=15, color="red", symbol="star")
            )
            st.plotly_chart(fig_highlight, use_container_width=True)
        else:
            st.warning("Lemma not found in current filter.")

# --- Morphology ---
with morph_tab:
    st.subheader("Suffix & Prefix Explorer")
    top_n = st.slider("Top-N", 5, 50, 10)

    # Top suffixes
    suffix_counts = filtered_df["suffix_3"].value_counts().nlargest(top_n).reset_index()
    suffix_counts.columns = ["suffix_3", "count"]
    fig_suffix = px.bar(suffix_counts, x="suffix_3", y="count", title="Most Common Suffix-3")
    st.plotly_chart(fig_suffix, use_container_width=True)

    # Top prefixes
    prefix_counts = filtered_df["prefix_3"].value_counts().nlargest(top_n).reset_index()
    prefix_counts.columns = ["prefix_3", "count"]
    fig_prefix = px.bar(prefix_counts, x="prefix_3", y="count", title="Most Common Prefix-3")
    st.plotly_chart(fig_prefix, use_container_width=True)

# --- Length vs Article ---
with length_tab:
    st.subheader("Word Length vs Article")
    plot_type = st.radio("Choose Plot Type", ["Boxplot", "Violin"])
    if plot_type == "Boxplot":
        fig = px.box(filtered_df, x="article", y="length", color="article")
    else:
        fig = px.violin(filtered_df, x="article", y="length", color="article", box=True, points="all")
    st.plotly_chart(fig, use_container_width=True)

# --- Morphological Networks ---

# with network_tab:
#     st.subheader("Morphological Network of Prefixes & Suffixes")
#     freq_cutoff = st.slider("Minimum Frequency", 5, 50, 10)
#     top_suffixes = filtered_df["suffix_3"].value_counts()[lambda x: x >= freq_cutoff].index
#     top_prefixes = filtered_df["prefix_3"].value_counts()[lambda x: x >= freq_cutoff].index

#     G = nx.Graph()
#     for _, row in filtered_df.iterrows():
#         if row["prefix_3"] in top_prefixes and row["suffix_3"] in top_suffixes:
#             G.add_edge(row["prefix_3"], row["suffix_3"])

#     fig_network = go.Figure()
#     if G.number_of_edges() > 0:
#         pos = nx.spring_layout(G, k=0.5)

#         # edges
#         edge_x, edge_y = [], []
#         for edge in G.edges():
#             x0, y0 = pos[edge[0]]
#             x1, y1 = pos[edge[1]]
#             edge_x += [x0, x1, None]
#             edge_y += [y0, y1, None]

#         fig_network.add_trace(go.Scatter(
#             x=edge_x, y=edge_y,
#             mode="lines",
#             line=dict(width=1, color="gray"),
#             hoverinfo="none"
#         ))

#         # nodes
#         node_x, node_y, node_text = [], [], []
#         for node in G.nodes():
#             x, y = pos[node]
#             node_x.append(x)
#             node_y.append(y)
#             node_text.append(node)

#         fig_network.add_trace(go.Scatter(
#             x=node_x, y=node_y,
#             mode="markers+text",
#             text=node_text,
#             textposition="top center",
#             marker=dict(size=10, color="blue"),
#             hovertext=node_text
#         ))

#         fig_network.update_layout(
#             showlegend=False,
#             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
#         )

#     else:
#         fig_network.add_annotation(
#             text="No edges found for this frequency cutoff.",
#             showarrow=False
#         )

#     st.plotly_chart(fig_network, use_container_width=True)
    
        

# --- Word Spotlight ---
with spotlight_tab:
    st.subheader("Word Spotlight")
    word = st.text_input("Enter a word to see its similar meaning words(Semantic similarity)")
    k_neighbors = st.slider("Number of Nearest Neighbors", 1, 20, 5)
    if word in filtered_df["lemma"].values:
        row = filtered_df[filtered_df["lemma"] == word].iloc[0]
        st.write("**Lemma:**", row["lemma"])
        st.write("**Article:**", row["article"])
        st.write("**Length:**", row["length"])
        st.write("**Prefix-3:**", row["prefix_3"])
        st.write("**Suffix-3:**", row["suffix_3"])

        # Compute neighbors by Euclidean distance in projection space
        distances = ((filtered_df["projection_x"] - row["projection_x"])**2 + (filtered_df["projection_y"] - row["projection_y"])**2)**0.5
        neighbors = filtered_df.assign(distance=distances).sort_values("distance").iloc[1:k_neighbors+1]

        st.write("### Nearest Neighbors")
        st.dataframe(neighbors[["lemma", "article", "length", "distance"]])
    elif word:
        st.warning("Word not found in current filter.")

# --- Density View ---
# with density_tab:
#     st.subheader("Word Density Map")
#     bandwidth = st.slider("Kernel Density Bandwidth", 0.1, 2.0, 0.5)
#     fig_density = ff.create_2d_density(
#     filtered_df["projection_x"], filtered_df["projection_y"],
#     colorscale="Viridis", hist_color="rgb(200,200,200)", point_size=2
# )

#     st.plotly_chart(fig_density, use_container_width=True)
