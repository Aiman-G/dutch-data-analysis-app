import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.data_loader import load_data_noun_embeddings
from utils.features import add_basic_features
from utils.rules import tree_rules_to_dataframe, make_rules_human, merge_rules
from utils.data_loader import load_data_noun_embeddings

st.set_page_config(page_title="Dutch article rule extractor", layout="wide")

st.title("🧠 Dutch article rule extractor — Decision Tree → readable rules")
# st.markdown("""
# Upload a CSV with at least:
# - a column with the target article label (e.g. 'label' with values 'de', 'het', or 'both')
# - a column with the lemma (noun) called 'lemma'
# - other feature columns (or the app will create suffix and length features automatically)

# This app trains a decision tree and extracts readable rules (conditions → majority class + percentage).
# """)
# -------------------------
# 1. Load and preview data
# -------------------------
df = load_data_noun_embeddings()
df.drop(['projection_x', 'projection_y', 'neighbors'], axis=1, inplace=True, errors="ignore")
st.subheader("Preview of dataset (first 5 rows)")
st.dataframe(df.head())

label_col, lemma_col = "article", "lemma"

# -------------------------
# 2. Features
# -------------------------
df = add_basic_features(df, lemma_col)
exclude = {label_col, lemma_col, "lemma_str"}
feature_cols = [c for c in df.columns if c not in exclude]

# st.write("Features used (auto-detected):", feature_cols)
selected_features = st.sidebar.multiselect("Select features", feature_cols, default=feature_cols)

if not selected_features:
    st.error("Please select at least one feature.")
    st.stop()

X = pd.get_dummies(df[selected_features].astype(str), drop_first=True)
y = df[label_col].astype(str)

# -------------------------
# 3. Model settings
# -------------------------
st.sidebar.subheader("Model controls")
max_depth = st.sidebar.selectbox("Tree depth", [2,3,4,5,6], index=2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 3)
random_state = st.sidebar.number_input("random_state (seed)", 0, 99999, 42)
test_size = st.sidebar.slider("Test set fraction (%)", 5, 50, 20) / 100.0

fit_button = st.sidebar.button("🔄 Fit model", width="stretch", type="primary")

# -------------------------
# 4. Train & Evaluate
# -------------------------
if fit_button:
    with st.spinner("***Training decision tree...***", show_time=True, width="stretch"):
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

        # store everything needed for later rendering
        st.session_state.update({
            "clf": clf,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "X": X, "y": y,
            "cv_scores": cv_scores,
        })

# -------------------------
# 5. Render results (if model exists)
# -------------------------
if "clf" in st.session_state:
    clf = st.session_state["clf"]
    X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
    y_train, y_test = st.session_state["y_train"], st.session_state["y_test"]
    X, y = st.session_state["X"], st.session_state["y"]
    cv_scores = st.session_state["cv_scores"]

    # ---- evaluation ----
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    st.subheader("Model evaluation")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Test accuracy", f"{acc:.3f}")
        st.write("CV mean:", f"{cv_scores.mean():.3f}", "±", f"{cv_scores.std():.3f}")
        st.dataframe(pd.DataFrame(report).T)
    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(conf_mat, interpolation='nearest')
        ax.set_xticks(np.arange(len(clf.classes_)))
        ax.set_yticks(np.arange(len(clf.classes_)))
        ax.set_xticklabels(clf.classes_, rotation=45, ha="right")
        ax.set_yticklabels(clf.classes_)
        for i in range(len(clf.classes_)):
            for j in range(len(clf.classes_)):
                ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="w")
        st.pyplot(fig)

    # ---- tree visualization ----
    st.subheader("Decision tree visualization")
    fig2, ax2 = plt.subplots(figsize=(14, 6 + max_depth))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, fontsize=8, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

    buf_tree_img = io.BytesIO()
    fig2.savefig(buf_tree_img, format="png", dpi=200, bbox_inches="tight")
    buf_tree_img.seek(0)

    # ---- rules ----
    st.subheader("Extracted rules")
    df_rules = tree_rules_to_dataframe(clf, X.columns.tolist(), X, y, min_support=1)

    if df_rules.empty:
        st.warning("No rules extracted.")
    else:
        counts_df = pd.json_normalize(df_rules['class_counts']).fillna(0).astype(int)
        rules_display = pd.concat([df_rules.drop(columns=['class_counts']), counts_df], axis=1)
        #rules_display["rule_human"] = make_rules_human(rules_display)
        rules_display = make_rules_human(rules_display)
        merge_rules_df = merge_rules(rules_display)


        #st.dataframe(rules_display)
        st.dataframe(merge_rules_df)

        # st.markdown("### Learner-friendly rules")
        # st.dataframe(rules_display[["rule_human", "predicted_class", "predicted_pct", "support_pct"]])

        # charts
        st.markdown("""
            ### Rule charts  
            Each chart summarizes one extracted rule:  

            - **Predicted % (green)**: how accurate this rule is.  
            - **Support % (blue)**: how much of the data this rule applies to.  

            👉 A good rule has both a high predicted % and a decent support %.  
            """)
        
        for i, row in rules_display.head(8).iterrows():
            st.markdown(f"**Rule #{i+1}** — predict **{row['predicted_class']}** "
                        f"({row['predicted_pct']:.2%}) — support {row['n_support']} "
                        f"({row['support_pct']:.2%})")
            fig_r, axr = plt.subplots(figsize=(6, 1.2))
            vals = [row['predicted_pct'], row['support_pct']]
            labels = ["Predicted accuracy", "Data coverage"]
            colors = ["#4CAF50", "#2196F3"]

            bars = axr.barh(labels, vals, color=colors)

            # Add percentage text inside bars
            for bar, val in zip(bars, vals):
                axr.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.0%}", va='center', ha='left', fontsize=9)

            axr.set_xlim(0, 1.05)
            axr.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            plt.tight_layout()
            st.pyplot(fig_r)
        # ---- export ----
        st.subheader("Export rules & tree")
        csv_buf = io.BytesIO()
        rules_display.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        pdf_buf = make_pdf(rules_display, buf_tree_img)

        st.download_button("📥 Download rules (CSV)", data=csv_buf.getvalue(),
                           file_name="rules.csv", mime="text/csv")
        # st.download_button("📄 Download rules & tree (PDF)", data=pdf_buf.getvalue(),
        #                    file_name="rules_and_tree.pdf", mime="application/pdf")
