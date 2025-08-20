import streamlit as st
import requests
import tempfile
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Configuration basique
st.set_page_config(
    page_title="🇹🇳 Tunisia Constitution RAG",
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
    .rag-answer {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1e88e5;
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

def setup_rag_system(text):
    """Set up the RAG system with document ingestion, embeddings, and vector store"""
    # Document ingestion and chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings and create vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def create_rag_chain(retriever):
    """Create the RAG chain with prompt template and LLM"""
    # Setup LLM (using Ollama with a local model - fallback to a simple approach if not available)
    try:
        llm = Ollama(model="llama2")
    except:
        # Fallback to a simple LLM simulation if Ollama is not available
        class SimpleLLM:
            def invoke(self, prompt):
                return "This is a simulated LLM response. Install Ollama with a model like llama2 for real RAG functionality."
        llm = SimpleLLM()
    
    # Custom prompt template
    prompt_template = """
    You are an expert on the Tunisian Constitution. Use the following context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer in a clear and concise manner:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    st.markdown('<h1 class="header">🇹🇳 Tunisia Constitution RAG Explorer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About")
        st.info("""
        Search through Tunisia's Constitution using
        Retrieval-Augmented Generation (RAG) technology.
        
        **How it works:**
        1. Downloads the constitution PDF
        2. Extracts and chunks the text
        3. Creates embeddings and vector store
        4. Retrieves relevant context
        5. Generates answers with LLM
        
        **Examples:**
        - president powers
        - judges appointed
        - citizens rights
        - amend constitution
        """)
    
    # Initialize session state variables
    if 'constitution_text' not in st.session_state:
        st.session_state.constitution_text = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False
    
    # Téléchargement automatique et configuration RAG
    if not st.session_state.setup_complete:
        with st.spinner("📥 Downloading constitution and setting up RAG system..."):
            try:
                response = requests.get("https://www.constituteproject.org/constitution/Tunisia_2014.pdf", timeout=30)
                if response.status_code == 200:
                    st.session_state.constitution_text = extract_text_from_pdf(response.content)
                    
                    # Set up RAG system
                    st.session_state.retriever = setup_rag_system(st.session_state.constitution_text)
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    st.session_state.setup_complete = True
                    
                    st.success("✅ RAG system loaded successfully!")
                else:
                    st.error("❌ Could not download constitution")
            except Exception as e:
                st.error(f"❌ Setup failed: {str(e)}")
    
    # Zone de recherche
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    question = st.text_input("🔍 Ask about the constitution:", placeholder="Enter your question...")
    search_btn = st.button("🚀 Ask", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recherche et génération de réponse
    if search_btn and question:
        if st.session_state.setup_complete:
            with st.spinner("🤔 Thinking..."):
                # Get the answer from RAG chain
                answer = st.session_state.rag_chain.invoke(question)
                
                # Display the answer
                st.markdown(f"""
                <div class="rag-answer">
                    <h3>🤖 RAG Answer:</h3>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Also show the retrieved context
                with st.expander("📚 View retrieved context"):
                    retrieved_docs = st.session_state.retriever.get_relevant_documents(question)
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"""
                        <div class="result-card">
                            <b>Context {i+1}:</b><br>
                            {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("❌ RAG system not loaded properly")

if __name__ == "__main__":
    main()