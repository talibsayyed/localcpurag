# Local Document RAG System

## Overview
The Local Document RAG (Retrieval-Augmented Generation) System is a powerful document search and analysis tool that combines semantic search capabilities with OCR (Optical Character Recognition) to enable efficient searching across both PDF documents and images. Built with Python, it utilizes FAISS for similarity search and Streamlit for a user-friendly interface.

## Features
- **Multi-Format Support**: Process both PDF documents and images (PNG, JPG, JPEG, TIFF)
- **Advanced OCR**: Enhanced image processing for improved text extraction
- **Dual Search Modes**: 
  - Semantic Search: Find contextually similar content
  - Exact Match: Locate precise text matches
- **Real-time Processing**: Immediate indexing and search capabilities
- **Interactive UI**: User-friendly interface for document upload and search
- **Confidence Scoring**: OCR confidence metrics and relevance scores for search results

## Prerequisites
- Python 3.8 or higher
- Tesseract OCR engine
- Sufficient RAM for document processing (minimum 8GB recommended)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/local-document-rag.git
cd local-document-rag
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

#### Windows:
1. Download the Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. Add Tesseract to your system PATH
4. Update the Tesseract path in the code if necessary:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Linux:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS:
```bash
brew install tesseract
```

## Project Structure
```
local-document-rag/
├── app.py                     # Streamlit frontend
├── enhanced_faiss_index.py    # Core RAG implementation
├── requirements.txt           # Package dependencies
├── temp_uploads/             # Temporary storage for uploaded files
└── README.md                 # Project documentation
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
TESSERACT_PATH=/path/to/tesseract
CHUNK_SIZE=512                # Text chunk size for indexing
MAX_UPLOAD_SIZE=200          # Maximum file upload size in MB
```

### requirements.txt
```
streamlit==1.24.0
pillow==9.5.0
pytesseract==0.3.10
opencv-python==4.7.0.72
sentence-transformers==2.2.2
faiss-cpu==1.7.4
PyMuPDF==1.22.5
python-dotenv==1.0.0
tqdm==4.65.0
numpy>=1.22.0
```

## Usage

### Starting the Application
```bash
streamlit run app.py
```

### Using the Interface
1. **Document Upload**
   - Drag and drop files into the upload area
   - Supported formats: PDF, PNG, JPG, JPEG, TIFF
   - Maximum file size: 200MB per file

2. **Search**
   - Enter your search query in the text input
   - Choose search type:
     - Semantic Search: For finding contextually similar content
     - Exact Match: For finding specific text

3. **Results**
   - View search results with relevance scores
   - Examine OCR confidence levels
   - Expand result cards for detailed context

## Performance Optimization

### Memory Usage
- Large documents are processed in chunks
- Embeddings are computed efficiently using batching
- Temporary files are cleaned up automatically

### Search Performance
- FAISS index enables fast similarity search
- Results are cached for repeated queries
- Batch processing for multiple documents

## Troubleshooting

### Common Issues

1. **OCR Quality Issues**
   - Ensure images are clear and well-lit
   - Try adjusting the image enhancement parameters
   - Verify Tesseract installation

2. **Memory Errors**
   - Reduce chunk size in configuration
   - Process fewer documents simultaneously
   - Increase system swap space

3. **Search Performance**
   - Index fewer documents if search is slow
   - Adjust FAISS index parameters
   - Consider using GPU acceleration

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/)
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract)
- Sentence embeddings by [sentence-transformers](https://www.sbert.net/)

## Contact
Your Name - your.email@example.com
Project Link: https://github.com/yourusername/local-document-rag

---
Made with ❤️ by Talib Sayyed

