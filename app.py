import streamlit as st
import requests
import tempfile
import os
import time
from PyPDF2 import PdfReader

# Configuration basique
st.set_page_config(
    page_title="🇹🇳 Tunisia Constitution",
    page_icon="🇹🇳",
    layout="wide"
)

# Style simple
st.markdown("""
<style>
    .header {
        text-align: center;
        color: #e70013;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e70013;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_content):
    """Extraire le texte du PDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_content)
        tmp_path = tmp_file.name
    
    text = ""
    try:
        reader = PdfReader(tmp_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except:
        text = "Could not extract text from PDF"
    
    os.unlink(tmp_path)
    return text

def simple_search(text, question):
    """Recherche simple par mots-clés"""
    lines = text.split('\n')
    results = []
    
    for line in lines:
        if any(word.lower() in line.lower() for word in question.split()):
            if len(line.strip()) > 20:  # Éviter les lignes trop courtes
                results.append(line.strip())
    
    return results[:5]  # Maximum 5 résultats

def main():
    st.markdown('<h1 class="header">🇹🇳 Tunisia Constitution Explorer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About")
        st.info("""
        Search through Tunisia's Constitution
        using simple keyword matching.
        
        **Examples:**
        - president powers
        - judges appointed
        - citizens rights
        - amend constitution
        """)
    
    # Téléchargement automatique
    if 'constitution_text' not in st.session_state:
        with st.spinner("📥 Downloading constitution..."):
            try:
                response = requests.get("https://www.constituteproject.org/constitution/Tunisia_2014.pdf", timeout=30)
                if response.status_code == 200:
                    st.session_state.constitution_text = extract_text_from_pdf(response.content)
                    st.success("✅ Constitution loaded successfully!")
                else:
                    st.error("❌ Could not download constitution")
            except:
                st.error("❌ Download failed")
    
    # Zone de recherche
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    question = st.text_input("🔍 Search the constitution:", placeholder="Enter your question...")
    search_btn = st.button("🚀 Search", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recherche
    if search_btn and question:
        if 'constitution_text' in st.session_state:
            results = simple_search(st.session_state.constitution_text, question)
            
            if results:
                st.success(f"📊 Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <b>Result {i}:</b><br>
                        {result[:200]}{'...' if len(result) > 200 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("🤔 No results found. Try different keywords.")
        else:
            st.error("❌ Constitution not loaded")

if __name__ == "__main__":
    main()