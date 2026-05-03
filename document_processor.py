import os
import google.generativeai as genai
from PyPDF2 import PdfReader

# --- NEW: LANGCHAIN RAG STACK ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document # 

class EnhancedMultiFormatChatbot:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        print("[DEBUG] Initializing LangChain & Vector Store...")
        
        # 1. Setup Google Embeddings Model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )
        
        # 2. Setup Text Splitter (Chops docs into 1000-character chunks with a 200-char overlap to keep sentences intact)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        # 3. Vector Database
        self.vector_store = None
        self.current_file = ""

    def extract_text_from_pdf(self, path):
        """Extracts raw text from a PDF file."""
        text = ""
        try:
            with open(path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ''
        except Exception as e:
            print(f"[ERROR extracting PDF] {e}")
        return text

    def _process_text_into_chroma(self, text):
        """Core RAG logic: Splits text and saves it into the Vector Database"""
        if not text.strip():
            return
            
        print("[DEBUG] Splitting document into chunks...")
        chunks = self.text_splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        print(f"[DEBUG] Created {len(docs)} chunks. Embedding into ChromaDB...")
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
        print("[DEBUG] Vector database updated successfully!")

    def load_file(self, file_path):
        """Legacy compatibility method for app.py"""
        if not os.path.exists(file_path):
            return False
        self.current_file = os.path.basename(file_path)
        return True

    def load_combined_text(self, combined_text):
        """Triggers the Vector Database processing when files are uploaded"""
        if combined_text:
            self._process_text_into_chroma(combined_text)
            return True
        return False

    def ask_question(self, question):
        """Queries the Vector Database and generates a context-aware answer"""
        if not self.vector_store:
            return "Please upload a document first before asking questions."
            
        try:
            # 1. RETRIEVAL: Find the 4 most relevant chunks of text to the user's question
            print(f"[DEBUG] Searching Vector DB for: {question}")
            relevant_docs = self.vector_store.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 2. AUGMENTATION: Build a precise prompt using ONLY the retrieved context
            prompt = f"""
            You are a highly intelligent Document AI Assistant.
            Use ONLY the following CONTEXT to answer the user's QUESTION.
            If the answer is not contained in the context, explicitly say: "I cannot find the answer to that in the uploaded document." Do not guess.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {question}
            """
            
            # 3. GENERATION: Let Gemini create the final answer
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"[RAG ERROR] {e}")
            return f"An error occurred while generating the response: {str(e)}"