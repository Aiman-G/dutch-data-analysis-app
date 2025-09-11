import streamlit as st

st.set_page_config(
   page_title="Dutch Nouns Explorer – AI-powered Learning",
    page_icon="🇳🇱",
    layout="wide",
)

st.title("🇳🇱 Dutch Data Analysis – AI-powered Learning")
st.caption("Exploring Dutch language data interactively")

# --- CSS styling ---
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: darkred;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper function for styled text ---
def big_text(text: str):
    return f'<p class="big-font">{text}</p>'



with st.expander("**What is this app about?**", expanded=True):
    st.markdown("***This project started as a way to explore and model data related to the Dutch language.***")


with st.expander("**Why**?", expanded=True):
    st.markdown(""" ***A month ago I began studying Dutch. Very quickly I felt overwhelmed by
            all the exceptions in the rules—especially with the articles het and de. 
           It seemed to me that the exception is the rule. So I thought:
          why not use data to uncover patterns myself? Maybe I could find shortcuts or
        at least a clearer picture. When I studied Mandarin Chinese in the past, grammar was simple,
        but tones drove me mad. 
           I tried a similar approach with the Mandarin Chinese tones. 
           Now I’m doing the same for Dutch. Hopefully it won’t just help me, 
           but also anyone else learning the language. Sharing is multiplying.***"""
           )
with st.expander("**For Whom?**"):
    st.markdown("""***For anyone who learn or teach Dutch. In fact,
    I believe those whose dutch is better than 1/2 A1 level ( my current level) 
                they would benefit more.***""")

with st.expander("**Methodology**"):
    st.markdown( "The **nouns' data**:\n"
        "- 📥 The data was obtained from **Hugging Face**: `mc4-nl-cleaned-mircor`.\n"
        "- 🔎 **5,000 rows** were randomly selected and downloaded.\n"
        "- 🧩 Nouns and articles were extracted using **spaCy** (`nl_core_news_md` model).\n"
        "- ⚡ Semantic embeddings were locally computed with a **FastText** model (`cc.nl.300.bin`).\n"
        "- 🗺️ Embeddings were projected into **2D** using the **UMAP** algorithm.\n"
        "- 📖 The data (all extracted features) is based on the **lemma** of each noun, meaning the base dictionary form.\n"
        "   (e.g., *honden* → **hond**, *huisje* → **huis**)."
    )



with st.expander("**Contact**"):
    st.markdown("""***In case you saw something interesting in the data, or you have a suggestion, 
                you can drop me an email at: aymen.omg@gmail.com.***""")