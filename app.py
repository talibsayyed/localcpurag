import streamlit as st
from pathlib import Path
import logging
from typing import Optional
import sys
from enhanced_faiss_index import EnhancedFAISSIndex  # Assuming the original code is in this file

# Set up logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="LOCAL DOCUMENT RAG",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
    }
    .stTextInput, .stSelectbox {
        background-color: #2D2D2D;
    }
    .result-box {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .stButton button {
        width: 100%;
        background-color: #4A4A4A;
        color: white;
    }
    .file-uploader {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 4px;
        border: 1px dashed #666;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'indexer' not in st.session_state:
        st.session_state.indexer = None
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = set()

def create_indexer() -> Optional[EnhancedFAISSIndex]:
    """Create and return a FAISS indexer instance"""
    try:
        return EnhancedFAISSIndex()
    except Exception as e:
        st.error(f"Error initializing indexer: {str(e)}")
        return None

def handle_file_upload(uploaded_files):
    """Handle file upload and indexing"""
    if not uploaded_files:
        return

    # Create temporary directory if it doesn't exist
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    # Initialize indexer if not already done
    if st.session_state.indexer is None:
        st.session_state.indexer = create_indexer()

    if st.session_state.indexer is None:
        st.error("Failed to initialize indexer")
        return

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.indexed_files:
            # Save uploaded file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Index the file
                chunks = st.session_state.indexer.process_file(str(file_path))
                if chunks:
                    embeddings = st.session_state.indexer.encoder.encode([chunk.text for chunk in chunks])
                    st.session_state.indexer.index.add(embeddings)
                    st.session_state.indexer.chunks.extend(chunks)
                    st.session_state.indexed_files.add(uploaded_file.name)
                    st.success(f"Successfully indexed {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

def main():
    """Main Streamlit application"""
    init_session_state()

    # Title
    st.title("LOCAL DOCUMENT RAG BY TALIB SAYYED")

    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Search Query")
        
        # Query input
        query = st.text_input("Enter your search query:", key="search_query")
        
        # Search type selection
        st.write("Search Type:")
        search_type = st.radio(
            "Select search type:",
            ["Semantic Search", "Exact Match"],
            horizontal=True,
            key="search_type"
        )

        # Search button
        if st.button("Search", key="search_button"):
            if not query:
                st.warning("Please enter a search query")
                return

            if st.session_state.indexer is None:
                st.warning("Please upload and index some documents first")
                return

            exact_match_only = search_type == "Exact Match"
            results = st.session_state.indexer.search(query, exact_match_only=exact_match_only)

            if not results:
                st.info("No results found")
            else:
                st.session_state.search_results = results

    with col2:
        st.subheader("Search Status")
        if 'search_results' in st.session_state:
            st.write("Search completed")
            
            st.subheader("Retrieved Text with Similarity Scores")
            for result in st.session_state.search_results:
                with st.expander(f"Result from {result['file_name']} (Page {result['page_number']})"):
                    st.write(f"Relevance Score: {result['relevance_score']:.3f}")
                    st.write(f"OCR Confidence: {result['confidence']:.1f}%")
                    st.write("Text:")
                    st.write(result['text'])

    # Document upload section at the bottom
    st.subheader("Document Upload")
    st.write("Upload documents (PDF/Images)")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        handle_file_upload(uploaded_files)

    # Display indexed files
    if st.session_state.indexed_files:
        st.write("Indexed Files:")
        for file_name in st.session_state.indexed_files:
            st.write(f"- {file_name}")

if __name__ == "__main__":
    main()