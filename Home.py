import streamlit as st

st.set_page_config(
   page_title="Dutch Language Explorer – AI-powered Learning",
   page_icon="🇳🇱",
   layout="wide",
)

st.title("🇳🇱 Dutch Language Explorer – AI-powered Learning")
st.caption("Discovering Dutch through data, AI, and interactive exploration")


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



# with st.expander("**What is this app about?**", expanded=True):
#     st.markdown("""
#     🌍 **Learning a language shouldn’t feel like memorizing endless rules.**  
#     Instead of reading grammar books cover to cover, why not let **data analysis** 
#     and **artificial intelligence (AI)** reveal the hidden patterns for us?  

#     This project reimagines how we explore and learn Dutch.  
#     By combining **linguistics** with **modern AI techniques**, the app makes the 
#     structure of Dutch  **interactive, visual, and fun** to explore.  
#     """)
with st.expander("**What is this app about?**", expanded=False):
    st.markdown("""
    🌍 **Learning Dutch shouldn’t just mean memorizing endless rules.**  
    Instead, we can use **data analysis** and **artificial intelligence (AI)** to reveal 
    the hidden structure of the language.  

    This project reimagines how we **learn and teach Dutch** by turning language data 
    into **interactive visualizations**.  
    - For now, the focus is on **nouns and articles (de/het)**.  
    - Soon, it will expand to cover **verbs, conjugations, and more**.  

    The goal: to make Dutch grammar **intuitive, visual, and data-driven**.
    """)




with st.expander("**Why?**", expanded=False):
    st.markdown("""
    ✨ **The personal spark:**  
    When I started learning Dutch ( a month ago), I was quickly overwhelmed by all the so-called "exceptions"—especially with the articles **de** and **het**.  
    It felt like the exception *was* the rule.  

    Instead of giving up, I thought:  
    💡 *What if I used data to uncover patterns myself?*  

    I had already experimented with this idea while learning **Mandarin Chinese**:  
    - In Chinese, grammar was simple but **tones** drove me mad.  
    - I used data to look for structure, which helped me learn more efficiently.  

    Now, I’m applying the same idea to Dutch.  
    Hopefully, this project helps not only me—but anyone curious about 
    learning Dutch in a smarter, data-driven way.  
    """)

with st.expander("**For Whom?**", expanded=False):
    st.markdown("""
    👩‍🎓 **This app is for anyone learning or teaching Dutch.**  

    - Beginners can **see patterns** that make learning less overwhelming.  
    - Intermediate/advanced learners can **analyze grammar structures** in depth.  
    - Teachers can use it as an **interactive classroom aid**.  
    - Linguists and AI enthusiasts can explore **language data** in a new, visual format.  
    """)


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
    st.markdown("""
    💌 Found something interesting in the data? Have a suggestion?  
    I’d love to hear from you → **aymen.omg@gmail.com**  
    """)

with st.expander("**☕ Support the Project**"):
    st.markdown("""
        🚀 **This app is just the beginning.**  
        Right now, it focuses on Dutch nouns and articles, but the vision is much bigger:  
        - Adding **verbs, conjugations, and irregular patterns**  
        - Expanding the dataset for more accurate insights  
        - Building richer **interactive visualizations** for learners and teachers  

        To make that possible, I need to cover **compute resources, hosting, and development time**.  
        Streamlit’s free tier is limited, and running AI models costs resources.  

        If you find this project useful and want to see it grow, you can support in two ways:  

        - 💡 **Share ideas or feedback** → Every suggestion helps improve the tool, data, or review the data.  
        - ☕ **Buy me a coffee** → A small donation helps cover compute + hosting costs.  

        👉 [Buy me a coffee](https://buymeacoffee.com/selenophil)  

        Thank you. Every bit of support helps keep this project alive and evolving.  
        """)

