import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import re
from collections import defaultdict

def tree_rules_to_dataframe(clf, feature_names, X_full, y_full, min_support=1):
    """
    Extract rules (paths to leaves) and compute support + class distribution.
    """
    tree = clf.tree_
    leaf_ids = clf.apply(X_full)
    leaves = np.unique(leaf_ids)
    rows = []

    for leaf in leaves:
        mask = (leaf_ids == leaf)
        n_support = mask.sum()
        if n_support < min_support:
            continue

        subset_y = y_full.iloc[mask]
        class_counts = subset_y.value_counts().to_dict()
        pred_class = subset_y.mode().iloc[0]
        pred_pct = class_counts.get(pred_class, 0) / n_support
        support_pct = n_support / len(X_full)

        sample_array = X_full.iloc[[np.nonzero(mask)[0][0]]]
        node_indicator = clf.decision_path(sample_array)
        path_nodes = node_indicator.indices.tolist()

        rule_conditions = []
        for nd in path_nodes:
            if tree.feature[nd] != -2:
                feat = feature_names[tree.feature[nd]]
                threshold = tree.threshold[nd]
                sample_val = sample_array.iloc[0].get(feat)
                cond = f"({feat} <= {threshold:.3f})" if sample_val <= threshold else f"({feat} > {threshold:.3f})"
                rule_conditions.append(cond)

        rule_str = " AND ".join(rule_conditions) if rule_conditions else "(root)"
        rows.append({
            "leaf_id": int(leaf),
            "rule": rule_str,
            "n_support": int(n_support),
            "support_pct": float(support_pct),
            "predicted_class": str(pred_class),
            "predicted_pct": float(pred_pct),
            "class_counts": class_counts
        })

    df_rules = pd.DataFrame(rows)
    df_rules = df_rules.sort_values("n_support", ascending=False).reset_index(drop=True)
    return df_rules


# def make_rules_human(df_rules: pd.DataFrame):
#     """
#     Convert technical rules into simplified, learner-friendly rules.
#     """
#     human_rules = []
#     for r in df_rules.itertuples():
#         text = r.rule
#         text = text.replace("length", "noun length")
#         text = text.replace("suffix_1", "last letter")
#         text = text.replace("suffix_2", "last 2 letters")
#         text = text.replace("suffix_3", "last 3 letters")
#         text = text.replace("suffix_4", "last 4 letters")
#         text = text.replace(" <= ", " is less or equal to ")
#         text = text.replace(" > ", " is greater than ")
#         human_rules.append(text)
#     return human_rules



def make_rules_human(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert machine-like rules into natural language explanations
    and add a 'rule_human' column.
    """

    human_rules = []

    for _, row in rules_df.iterrows():
        raw_rule = row["rule"]
        predicted_class = row["predicted_class"]
        pred_pct = row["predicted_pct"]
        n_samples = row.get("n_samples", None)
        support_pct = row.get("support_pct", None)

        conditions = []
        for cond in raw_rule.split(" AND "):
            cond = cond.strip("() ")

            # suffix
            if cond.startswith("suffix_"):
                m = re.match(r"suffix_\d+_(\w+)\s*(<=|>)\s*0\.500", cond)
                if m:
                    suf, op = m.groups()
                    if op == ">":
                        conditions.append(f"word ends with '{suf}'")
                    else:
                        conditions.append(f"word does not end with '{suf}'")

            # prefix
            elif cond.startswith("prefix_"):
                m = re.match(r"prefix_\d+_(\w+)\s*(<=|>)\s*0\.500", cond)
                if m:
                    pre, op = m.groups()
                    if op == ">":
                        conditions.append(f"word starts with '{pre}'")
                    else:
                        conditions.append(f"word does not start with '{pre}'")

            # length
            elif cond.startswith("length_"):
                m = re.match(r"length_(\d+)\s*(<=|>)\s*0\.500", cond)
                if m:
                    n, op = m.groups()
                    if op == ">":
                        conditions.append(f"word has length {n}")
                    else:
                        conditions.append(f"word does not have length {n}")

            # fallback
            else:
                conditions.append(cond)

        cond_str = " AND ".join(conditions) if conditions else "(always true)"

        # Build sentence
        human_sentence = f"If {cond_str} → predict **{predicted_class}** ({pred_pct:.0%} accurate"
        if n_samples is not None:
            human_sentence += f", covers {n_samples} nouns"
        if support_pct is not None:
            human_sentence += f", {support_pct:.1%} of data"
        human_sentence += ")."

        human_rules.append(human_sentence)

    rules_df["rule_human"] = human_rules
    return rules_df


def merge_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rules that share the same predicted class and accuracy/support 
    into a single OR condition if possible.
    """
    merged = []
    grouped = defaultdict(list)

    # group by prediction + accuracy + support
    for _, row in rules_df.iterrows():
        key = (row["predicted_class"], round(row["predicted_pct"], 2), round(row["support_pct"], 3))
        grouped[key].append(row)

    for (cls, acc, sup), rows in grouped.items():
        if len(rows) == 1:
            merged.append(rows[0])
        else:
            # try to OR their conditions
            conds = [r["rule_human"].split(" → ")[0][3:] for _, r in enumerate(rows)]
            or_rule = " OR ".join([c.replace("If ", "") for c in conds])
            merged_text = f"If {or_rule} → predict **{cls}** ({acc:.0%} accurate, {sup:.1%} of data)."

            first_row = rows[0].copy()
            first_row["rule_human"] = merged_text
            merged.append(first_row)

    return pd.DataFrame(merged)


def make_pdf(rules_df: pd.DataFrame, tree_img_buf: io.BytesIO) -> io.BytesIO:
    """
    Create a PDF report with extracted rules and decision tree image.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Dutch Article Rule Extractor — Rules Export")
    c.setFont("Helvetica", 9)
    y -= 18
    c.drawString(margin, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 20

    # Top rules
    topn = min(20, len(rules_df))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, f"Top {topn} rules (by support):")
    y -= 16
    c.setFont("Helvetica", 9)

    for i in range(topn):
        if y < 120:
            c.showPage()
            y = height - margin
        row = rules_df.iloc[i]
        line = f"{i+1}. [{row['n_support']} obs, {row['support_pct']:.1%}] → {row['predicted_class']} ({row['predicted_pct']:.1%}) :: {row['rule']}"
        while len(line) > 120:
            c.drawString(margin, y, line[:120])
            y -= 12
            line = line[120:]
        c.drawString(margin, y, line)
        y -= 12

    # Add tree image
    c.showPage()
    tree_img_buf.seek(0)
    img = ImageReader(tree_img_buf)
    img_w, img_h = img.getSize()
    aspect = img_h / img_w
    target_w = width - 2*margin
    target_h = target_w * aspect
    if target_h > height - 2*margin:
        target_h = height - 2*margin
        target_w = target_h / aspect
    c.drawImage(img, margin, height - margin - target_h, width=target_w, height=target_h)

    c.save()
    buf.seek(0)
    return buf
