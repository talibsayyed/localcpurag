import os
from pathlib import Path
import logging
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from tqdm import tqdm
from PIL import Image
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ------ Data Classes -----------------------
@dataclass
class TextChunk:
    """
    Data class to store metadata and content of text chunks
    """
    text: str
    file_name: str
    page_number: int
    chunk_index: int
    confidence: float = 0.0  # Added confidence field
# ------------------------------------------

# ------ Enhanced FAISS Index Class --------
class EnhancedFAISSIndex:
    """
    Class to create and manage a searchable FAISS index from document data
    """
    def __init__(self, chunk_size: int = 512):
        # ------ Initialization ----------------
        logging.info("Initializing Enhanced FAISS indexer...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks: List[TextChunk] = []
        self.chunk_size = chunk_size
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        self.document_extensions = {'.pdf'}
        # --------------------------------------

    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results across different font styles
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced)
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced)

    def post_process_ocr_text(self, text: str) -> str:
        """
        Clean up OCR output text
        """
        # Remove excess whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        replacements = {
            '|': 'I',  # Vertical bar to I
            '0': 'O',  # Zero to O in likely letter contexts
            'l': 'I',  # lowercase L to I in likely contexts
            'rn': 'm',  # Common misrecognition
            'cl': 'd',  # Common misrecognition
        }
        
        # Apply replacements in context
        words = text.split()
        for i, word in enumerate(words):
            if len(word) == 1:  # Single character words
                if word in replacements:
                    words[i] = replacements[word]
            else:
                # Apply multi-character replacements
                for old, new in replacements.items():
                    if len(old) > 1:
                        words[i] = words[i].replace(old, new)
                
        text = ' '.join(words)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text

    def extract_text_from_image(self, image_path: str) -> tuple[str, float]:
        """
        Extract text from an image using enhanced preprocessing and multiple OCR passes
        Returns tuple of (text, confidence)
        """
        try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            # Enhanced preprocessing
            enhanced_image = self.enhance_image_for_ocr(image)
            
            # Multiple OCR passes with different configurations
            text_results = []
            
            # Standard configuration
            text_results.append(
                pytesseract.image_to_string(
                    enhanced_image,
                    config='--psm 3 --oem 3'
                )
            )
            
            # Configuration optimized for sparse text
            text_results.append(
                pytesseract.image_to_string(
                    enhanced_image,
                    config='--psm 6 --oem 3'
                )
            )
            
            # Configuration with character whitelist
            text_results.append(
                pytesseract.image_to_string(
                    enhanced_image,
                    config='--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?()-:;"\'$ '
                )
            )
            
            # Select best result based on confidence
            confidences = []
            for text in text_results:
                try:
                    conf_data = pytesseract.image_to_data(enhanced_image, output_type=pytesseract.Output.DICT)
                    avg_conf = sum(conf_data['conf']) / len(conf_data['conf'])
                    confidences.append(avg_conf)
                except:
                    confidences.append(0)
            
            best_idx = confidences.index(max(confidences))
            best_text = text_results[best_idx]
            best_confidence = confidences[best_idx]
            
            # Post-processing
            best_text = self.post_process_ocr_text(best_text)
            
            return best_text, best_confidence
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return "", 0.0

    # ------ Text Chunking -------------------
    def chunk_text(self, text: str, file_name: str, page_number: int, confidence: float = 1.0) -> List[TextChunk]:
        """
        Split text into manageable chunks with associated metadata
        """
        text = ' '.join(text.split())
        if not text:
            return []

        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for word in words:
            if current_length + len(word) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(
                        TextChunk(
                            text=' '.join(current_chunk),
                            file_name=file_name,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            confidence=confidence
                        )
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0

            current_chunk.append(word)
            current_length += len(word) + 1

        if current_chunk:
            chunks.append(
                TextChunk(
                    text=' '.join(current_chunk),
                    file_name=file_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    confidence=confidence
                )
            )

        return chunks

    def extract_text_from_pdf(self, pdf_path: str) -> List[tuple[str, float]]:
        """
        Extract text from all pages of a PDF
        Returns list of tuples (text, confidence)
        """
        try:
            doc = fitz.open(pdf_path)
            results = []
            for page in doc:
                text = page.get_text()
                # Assume high confidence for PDF text extraction
                results.append((text, 1.0))
            doc.close()
            return results
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def process_file(self, file_path: str) -> List[TextChunk]:
        """
        Process a file and convert its content into text chunks
        """
        file_extension = Path(file_path).suffix.lower()
        chunks = []

        try:
            if file_extension in self.image_extensions:
                text, confidence = self.extract_text_from_image(file_path)
                if text.strip():
                    chunks.extend(self.chunk_text(text, file_path, 1, confidence))
            elif file_extension in self.document_extensions:
                pages = self.extract_text_from_pdf(file_path)
                for page_num, (page_text, confidence) in enumerate(pages, 1):
                    if page_text.strip():
                        chunks.extend(self.chunk_text(page_text, file_path, page_num, confidence))
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

        return chunks

    def index_documents(self, folder_path: str) -> None:
        """
        Index documents from a folder into the FAISS index
        """
        logging.info(f"Starting document indexing from {folder_path}")
        supported_extensions = self.image_extensions.union(self.document_extensions)
        files = [file for ext in supported_extensions for file in Path(folder_path).rglob(f'*{ext}')]

        for file_path in tqdm(files):
            try:
                chunks = self.process_file(str(file_path))
                if chunks:
                    embeddings = self.encoder.encode([chunk.text for chunk in chunks])
                    self.index.add(np.array(embeddings))
                    self.chunks.extend(chunks)
                    logging.info(f"Indexed {file_path}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")

    def find_exact_matches(self, query: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Find exact matches for a query in indexed chunks
        """
        results = []
        query = query if case_sensitive else query.lower()
        pattern = r'\b' + re.escape(query) + r'\b'

        for chunk in self.chunks:
            text_to_search = chunk.text if case_sensitive else chunk.text.lower()
            matches = list(re.finditer(pattern, text_to_search))
            for match in matches:
                context_start = max(0, match.start() - 50)
                context_end = min(len(text_to_search), match.end() + 50)
                results.append({
                    'text': chunk.text[context_start:context_end],
                    'file_name': chunk.file_name,
                    'page_number': chunk.page_number,
                    'match_position': (match.start(), match.end()),
                    'is_exact_match': True,
                    'relevance_score': chunk.confidence,  # Use OCR confidence for relevance
                    'confidence': chunk.confidence
                })

        return results

    def search(self, query: str, top_k: int = 5, exact_match_only: bool = False) -> List[Dict]:
        """
        Search the FAISS index with optional exact match filtering
        """
        if not self.chunks:
            return []

        exact_results = self.find_exact_matches(query)
        if exact_match_only:
            return sorted(exact_results, key=lambda x: x['confidence'], reverse=True)[:top_k]

        if not exact_results:
            query_embedding = self.encoder.encode([query])
            distances, indices = self.index.search(query_embedding, top_k)
            return [
                {
                    'text': self.chunks[idx].text,
                    'file_name': self.chunks[idx].file_name,
                    'page_number': self.chunks[idx].page_number,
                    'relevance_score': 1 / (1 + distances[0][i]),
                    'is_exact_match': False,
                    'confidence': self.chunks[idx].confidence
                }
                for i, idx in enumerate(indices[0]) if idx < len(self.chunks)
            ]

        return sorted(exact_results, key=lambda x: x['confidence'], reverse=True)[:top_k]

def main():
    """
    Main function to run the document indexing and search utility
    """
    logging.basicConfig(level=logging.INFO)
    indexer = EnhancedFAISSIndex()
    
    indexer.index_documents("MVF-Test-Files")

    print("\nIndexing complete. Ready to accept queries.")
    print("You can input one query at a time. Type 'quit' to exit.")

    while True:
        try:
            query = input("\nEnter your search query (or 'quit' to exit): ").strip()
            if query.lower() == 'quit':
                print("\nExiting the search utility. Goodbye!")
                break

            search_type = input("Search type (1: Exact only, 2: Both exact and semantic): ").strip()
            exact_match_only = search_type == "1"

            results = indexer.search(query, exact_match_only=exact_match_only)

            if not results:
                print("\nNo results found.")
                continue

            print("\nSearch Results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. File: {result['file_name']}")
                print(f"   Page: {result['page_number']}")
                print(f"   Text: {result['text'][:200]}...")
                print(f"   Match Type: {'Exact' if result.get('is_exact_match') else 'Semantic'}")
                print(f"   Relevance: {result['relevance_score']:.3f}")
                print(f"   OCR Confidence: {result['confidence']:.1f}%")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting the search utility. Goodbye!")
            break

if __name__ == "__main__":
    main()