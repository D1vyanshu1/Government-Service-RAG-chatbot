import os
from pypdf import PdfReader
from typing import List

class PDFLoader:
    
    def __init__(self, pdf_dir: str = "data/pdfs", chunk_size: int = 512):
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
    
    def load_all_pdfs(self) -> List[dict]:
        documents = []
        
        for pdf_file in os.listdir(self.pdf_dir):
            if not pdf_file.endswith('.pdf'):
                continue
            
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    for chunk in self._chunk_text(text):
                        documents.append({
                            "doc_name": pdf_file,
                            "page": page_num + 1,
                            "text": chunk
                        })
            except Exception as e:
                print(f"Warning: Could not read {pdf_file}: {e}")
                continue
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = ""
        
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if len(c) > 50]