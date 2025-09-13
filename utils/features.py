
import pandas as pd

def add_basic_features(df_in: pd.DataFrame, lemma_col: str) -> pd.DataFrame:
    """
    Add basic features: lemma_str, word length, suffixes (1–4 chars).
    """
    df = df_in.copy()
    df['lemma_str'] = df[lemma_col].astype(str)
    #df['len_lemma'] = df['lemma_str'].str.len()
    

    # Add suffix features
    # for k in (1, 2, 3, 4):
    #     df[f'suf_{k}'] = df['lemma_str'].str[-k:].fillna('').astype(str)
    
    df[f'suffix_1'] = df['lemma_str'].str[-1:].fillna('').astype(str)


    return df
