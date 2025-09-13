import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming your utility functions are in a 'utils' folder
from utils.data_loader import load_data_noun_embeddings
from utils.features import add_basic_features
from utils.rules import tree_rules_to_dataframe, make_rules_human, merge_rules

st.set_page_config(page_title="Dutch article rule extractor", layout="wide")

st.title("🧠 Dutch article rule extractor — Decision Tree → readable rules")

# --- CACHE 1: Initial Data Loading & Processing ---
@st.cache_data
def get_and_prepare_data():
    """
    Loads and performs initial processing. Runs ONLY ONCE.
    """
    df_loaded = load_data_noun_embeddings()
    df_loaded.drop(['projection_x', 'projection_y', 'neighbors'], axis=1, inplace=True, errors="ignore")
    df_processed = add_basic_features(df_loaded, "lemma")
    return df_processed

# --- CACHE 2: Final Model Input Creation ---
@st.cache_data
def create_model_inputs(df, selected_features_tuple):
    """
    Performs one-hot encoding. Reruns ONLY if selected_features_tuple changes.
    """
    # --- THIS IS THE FIX ---
    # Convert the incoming tuple back to a list for pandas column selection
    selected_features_list = list(selected_features_tuple)
    
    X = pd.get_dummies(df[selected_features_list].astype(str), drop_first=True)
    y = df["article"].astype(str)
    return X, y

# --- Load and Process Data using Cached Functions ---
df = get_and_prepare_data()

st.subheader("Preview of dataset (first 5 rows)")
st.dataframe(df.head())

# --- Feature Selection (Sidebar) ---
st.sidebar.subheader("Feature Selection")
exclude = {"article", "lemma", "lemma_str"}
all_feature_cols = [c for c in df.columns if c not in exclude]

# We create a tuple so the selection is "hashable" and can be cached
selected_features = tuple(st.sidebar.multiselect("Select features", all_feature_cols, default=all_feature_cols))

if not selected_features:
    st.error("Please select at least one feature.")
    st.stop()

# Create model inputs using the second cached function
X, y = create_model_inputs(df, selected_features)

# --- Model Settings (Sidebar) ---
st.sidebar.subheader("Model controls")
max_depth = st.sidebar.selectbox("Tree depth", [2,3,4,5,6], index=2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 3)
random_state = st.sidebar.number_input("random_state (seed)", 0, 99999, 42)
test_size = st.sidebar.slider("Test set fraction (%)", 5, 50, 20) / 100.0

fit_button = st.sidebar.button("🔄 Fit model", type="primary")

# --- Train & Evaluate ---
if fit_button:
    with st.spinner("Training decision tree..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        ).fit(X_train, y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        st.session_state.clf_results = {
            "clf": clf, "X_test": X_test, "y_test": y_test,
            "X": X, "y": y, "cv_scores": cv_scores,
        }

# --- Render Results ---
if "clf_results" in st.session_state:
    results = st.session_state.clf_results
    clf = results["clf"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    X_full = results["X"]
    y_full = results["y"]
    cv_scores = results["cv_scores"]

    st.subheader("Model evaluation")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Test accuracy", f"{acc:.3f}")
        st.write(f"5-fold CV mean accuracy: **{cv_scores.mean():.3f}** (± {cv_scores.std():.3f})")
        st.dataframe(pd.DataFrame(report).T.style.format("{:.2f}"))
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(len(clf.classes_)))
        ax.set_yticks(np.arange(len(clf.classes_)))
        ax.set_xticklabels(clf.classes_, rotation=45, ha="right")
        ax.set_yticklabels(clf.classes_)
        for i in range(len(clf.classes_)):
            for j in range(len(clf.classes_)):
                color = "white" if conf_mat[i, j] > conf_mat.max() / 2. else "black"
                ax.text(j, i, conf_mat[i, j], ha="center", va="center", color=color)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Decision tree visualization")
    fig2, ax2 = plt.subplots(figsize=(14, 6 + max_depth))
    plot_tree(clf, feature_names=X_full.columns.tolist(), class_names=clf.classes_, filled=True, fontsize=8, ax=ax2)
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Extracted rules")
    df_rules = tree_rules_to_dataframe(clf, X_full.columns.tolist(), X_full, y_full, min_support=1)
    
    if not df_rules.empty:
        counts_df = pd.json_normalize(df_rules['class_counts']).fillna(0).astype(int)
        rules_display = pd.concat([df_rules.drop(columns=['class_counts']), counts_df], axis=1)
        rules_display = make_rules_human(rules_display)
        merge_rules_df = merge_rules(rules_display)
        st.dataframe(merge_rules_df)

        st.markdown("### Rule charts")
        for i, row in rules_display.head(8).iterrows():
            st.markdown(f"**Rule #{i+1}**: Predicts **{row['predicted_class']}**")
            fig_r, axr = plt.subplots(figsize=(6, 1.2))
            vals = [row['predicted_pct'], row['support_pct']]
            labels = ["Predicted accuracy", "Data coverage"]
            colors = ["#4CAF50", "#2196F3"]
            bars = axr.barh(labels, vals, color=colors)
            for bar, val in zip(bars, vals):
                axr.text(val, bar.get_y() + bar.get_height()/2, f" {val:.0%}", va='center', ha='left', fontsize=9)
            axr.set_xlim(0, 1.05)
            axr.set_xticks([])
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)