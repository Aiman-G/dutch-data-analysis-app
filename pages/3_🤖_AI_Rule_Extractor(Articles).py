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

# --- 1. Cached Data Loading and Preparation ---
@st.cache_data(show_spinner=False, max_entries=1, ttl=3600)
def get_and_prepare_data():
    """
    Loads the initial dataset and performs basic feature engineering.
    This function is cached to prevent reloading on every script rerun.
    """
    df_loaded = load_data_noun_embeddings()
    df_loaded.drop(['projection_x', 'projection_y', 'neighbors'], axis=1, inplace=True, errors="ignore")
    # Note: add_basic_features is now called only once inside the cached function
    df_processed = add_basic_features(df_loaded, "lemma")
    return df_processed

# --- Call the cached function to get the data ---
with st.spinner("Loading data..."):
    df = get_and_prepare_data()

st.subheader("Preview of dataset (first 5 rows)")
st.dataframe(df.head())

# Define column names
label_col, lemma_col = "article", "lemma"

# --- 2. Feature Selection in Sidebar ---
st.sidebar.subheader("Feature Selection")
exclude = {label_col, lemma_col, "lemma_str"}
all_feature_cols = [c for c in df.columns if c not in exclude]
selected_features = st.sidebar.multiselect("Select features", all_feature_cols, default=all_feature_cols)

if not selected_features:
    st.error("Please select at least one feature.")
    st.stop()

# --- 3. Model settings ---
st.sidebar.subheader("Model controls")
max_depth = st.sidebar.selectbox("Tree depth", [2,3,4,5,6], index=2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 3)
random_state = st.sidebar.number_input("random_state (seed)", 0, 99999, 42)
test_size = st.sidebar.slider("Test set fraction (%)", 5, 50, 20) / 100.0

fit_button = st.sidebar.button("🔄 Fit model", type="primary")

# --- 4. Train & Evaluate ---
if fit_button:
    with st.spinner("Training decision tree..."):
        # Perform one-hot encoding on the user-selected features
        X = pd.get_dummies(df[selected_features].astype(str), drop_first=True)
        y = df[label_col].astype(str)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        ).fit(X_train, y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        # We can run CV on the full dataset to get a more robust estimate
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        # Store only essential results in session state
        st.session_state.clf_results = {
            "clf": clf,
            "X_test": X_test, 
            "y_test": y_test,
            "cv_scores": cv_scores,
            "feature_names": X.columns.tolist(),
            "class_names": clf.classes_,
            "selected_features": selected_features
        }
        
        # Clear large objects to free memory
        del X, X_train, y_train
        import gc
        gc.collect()

# --- 5. Render results (if model exists) ---
if "clf_results" in st.session_state:
    results = st.session_state.clf_results
    clf, X_test, y_test, cv_scores = (
        results["clf"], results["X_test"], results["y_test"], results["cv_scores"]
    )
    feature_names = results["feature_names"]
    class_names = results["class_names"]
    selected_features = results["selected_features"]

    # ---- Evaluation ----
    st.subheader("Model evaluation")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=class_names)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Test accuracy", f"{acc:.3f}")
        st.write(f"5-fold CV mean accuracy: **{cv_scores.mean():.3f}** (± {cv_scores.std():.3f})")
        st.dataframe(pd.DataFrame(report).T.style.format("{:.2f}"))
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                color = "white" if conf_mat[i, j] > conf_mat.max() / 2. else "black"
                ax.text(j, i, conf_mat[i, j], ha="center", va="center", color=color)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)  # Clear figure after rendering
        plt.close(fig)

    # ---- Tree Visualization ----
    st.subheader("Decision tree visualization")
    # Only show tree if depth is not too large
    if max_depth <= 5:
        fig2, ax2 = plt.subplots(figsize=(14, 6 + max_depth))
        plot_tree(clf, feature_names=feature_names, class_names=class_names, 
                 filled=True, fontsize=8, ax=ax2)
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
    else:
        st.warning("Tree visualization is disabled for depths > 5 to save memory.")

    # ---- Rules Extraction ----
    st.subheader("Extracted rules")
    
    # Recreate X for rule extraction but with optimized memory usage
    with st.spinner("Extracting rules..."):
        X_rules = pd.get_dummies(df[selected_features].astype(str), drop_first=True)
        y_rules = df[label_col].astype(str)
        
        df_rules = tree_rules_to_dataframe(clf, feature_names, X_rules, y_rules, min_support=1)
        
        # Free memory after rule extraction
        del X_rules, y_rules
        import gc
        gc.collect()

    if not df_rules.empty:
        counts_df = pd.json_normalize(df_rules['class_counts']).fillna(0).astype(int)
        rules_display = pd.concat([df_rules.drop(columns=['class_counts']), counts_df], axis=1)
        rules_display = make_rules_human(rules_display)
        merge_rules_df = merge_rules(rules_display)
        
        st.dataframe(merge_rules_df)

        # ---- Charts Section ----
        st.markdown("### Rule charts")
        st.markdown("A good rule has high **Predicted accuracy** (green) and reasonable **Data coverage** (blue).")
        
        # Limit the number of rules shown to save memory
        max_rules_to_show = 5
        for i, row in rules_display.head(max_rules_to_show).iterrows():
            st.markdown(f"**Rule #{i+1}**: Predicts **{row['predicted_class']}** ({row['predicted_pct']:.1%}) "
                        f"with a data coverage of {row['support_pct']:.1%}")
            
            fig_r, axr = plt.subplots(figsize=(6, 1.2))
            
            vals = [row['predicted_pct'], row['support_pct']]
            labels = ["Predicted accuracy", "Data coverage"]
            colors = ["#4CAF50", "#2196F3"]
            bars = axr.barh(labels, vals, color=colors)
            for bar, val in zip(bars, vals):
                axr.text(val, bar.get_y() + bar.get_height()/2, f" {val:.0%}", 
                         va='center', ha='left', fontsize=9)

            axr.set_xlim(0, 1.05)
            axr.spines['top'].set_visible(False)
            axr.spines['right'].set_visible(False)
            axr.spines['bottom'].set_visible(False)
            axr.set_xticks([])
            plt.tight_layout()
            
            st.pyplot(fig_r, clear_figure=True)
            plt.close(fig_r)
            
        if len(rules_display) > max_rules_to_show:
            st.info(f"Showing first {max_rules_to_show} rules. Total rules: {len(rules_display)}")