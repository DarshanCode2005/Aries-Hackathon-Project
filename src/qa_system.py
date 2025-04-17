import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Tuple
import textwrap
from pathlib import Path
import pickle
import logging
from dotenv import load_dotenv
import unidecode
import re
import latex2mathml.converter
import markdown
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGQASystem:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        index_path: str = 'data/processed/faiss_index',
        chunks_path: str = 'data/processed/text_chunks.pkl'
    ):
        """
        Initialize the RAG-based Q&A system.
        
        Args:
            model_name (str): Name of the Sentence Transformer model to use
            index_path (str): Path to save/load the FAISS index
            chunks_path (str): Path to save/load the text chunks
        """
        self.model_name = model_name
        self.index_path = index_path
        self.chunks_path = chunks_path
        
        # Initialize Sentence Transformer
        try:
            self.encoder = SentenceTransformer(model_name)
            self.vector_dim = self.encoder.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model: {str(e)}")
            raise
            
        # Initialize FAISS index
        self.index = None
        self.text_chunks = []
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Successfully initialized Gemini model")
            except Exception as e:
                logger.error(f"Error initializing Gemini model: {str(e)}")
                self.gemini_model = None
        else:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            self.gemini_model = None
    
    def create_text_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks of approximately equal size.
        
        Args:
            text (str): Text to split
            chunk_size (int): Target size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # Clean the text first
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = ' '.join(text.split())   # Remove extra whitespace
        
        # Split into chunks while preserving word boundaries
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for the space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def build_index(self, texts: List[str]) -> None:
        """
        Build FAISS index from a list of texts.
        
        Args:
            texts (List[str]): List of texts to index
        """
        try:
            # Create chunks
            all_chunks = []
            for text in texts:
                chunks = self.create_text_chunks(text)
                all_chunks.extend(chunks)
            
            # Generate embeddings
            embeddings = self.encoder.encode(all_chunks, show_progress_bar=True)
            
            # Build FAISS index
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store text chunks
            self.text_chunks = all_chunks
            
            # Save index and chunks
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.text_chunks, f)
                
            logger.info(f"Built index with {len(all_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise
    
    def load_index(self) -> bool:
        """
        Load FAISS index and text chunks from disk.
        
        Returns:
            bool: True if loading was successful
        """
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, 'rb') as f:
                    self.text_chunks = pickle.load(f)
                logger.info(f"Loaded index with {len(self.text_chunks)} chunks")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve most similar text chunks for a query.
        
        Args:
            query (str): Query text
            k (int): Number of chunks to retrieve
            
        Returns:
            List[Tuple[str, float]]: List of (chunk, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode([query])[0]
            
            # Search index
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                k
            )
            
            # Get text chunks and scores
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.text_chunks):
                    results.append((self.text_chunks[idx], float(dist)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using Gemini API.
        
        Args:
            query (str): User's question
            context (str): Retrieved context
            
        Returns:
            str: Generated answer
        """
        if not self.gemini_model:
            return "Error: Gemini API key not configured"
            
        try:
            prompt = f"""You are a helpful research assistant. Using ONLY the provided context, answer the question. 
            If you cannot answer the question based on the context, explain why.
            
            Context:
            ```
            {context}
            ```
            
            Question: {query}
            
            Instructions:
            1. Use ONLY the information from the context above
            2. If the context doesn't contain relevant information, say so
            3. Provide specific details and cite from the context when possible
            4. Be concise but thorough
            
            Answer:"""
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def process_latex(self, text: str) -> str:
        """
        Convert LaTeX expressions to MathML for proper rendering.
        
        Args:
            text (str): Text containing LaTeX expressions
            
        Returns:
            str: Text with LaTeX converted to MathML
        """
        def replace_latex(match):
            latex = match.group(1)
            try:
                mathml = latex2mathml.converter.convert(latex)
                return mathml
            except:
                return match.group(0)
        
        # Convert inline math mode
        text = re.sub(r'\$([^$]+)\$', replace_latex, text)
        # Convert display math mode
        text = re.sub(r'\\\[(.*?)\\\]', replace_latex, text)
        return text
    
    def format_text_with_gemini(self, text: str) -> str:
        """
        Use Gemini to format the text properly with correct spacing and mathematical notation.
        
        Args:
            text (str): Text to format
            
        Returns:
            str: Formatted text
        """
        if not self.gemini_model:
            return text
            
        try:
            prompt = f"""Format the following text to have proper spacing between words and proper mathematical notation. 
            Preserve all LaTeX expressions (content between $ signs or \\[ \\]) exactly as they are.
            Preserve all mathematical symbols.
            Return ONLY the formatted text, no explanations.
            
            Text to format:
            {text}
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0,
                    'top_p': 1,
                    'top_k': 1
                }
            )
            
            formatted_text = response.text.strip()
            return formatted_text
        except Exception as e:
            logger.warning(f"Error formatting text with Gemini: {str(e)}")
            return text
    
    def render_markdown_and_latex(self, text: str) -> str:
        """
        Render both Markdown and LaTeX in the text.
        
        Args:
            text (str): Text to render
            
        Returns:
            str: Rendered HTML
        """
        # First process LaTeX
        text_with_mathml = self.process_latex(text)
        
        # Convert markdown to HTML
        html = markdown.markdown(text_with_mathml, extensions=['extra'])
        
        # Clean up the HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Add MathJax for any remaining LaTeX
        html = str(soup)
        return html

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text using unidecode and basic cleaning.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Basic cleaning first
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # Use unidecode for initial normalization while preserving some special characters
        # Save mathematical symbols before unidecode
        math_symbols = re.findall(r'[∈≤∼∀∃∇∫∑∏⊆⊂∪∩×±→∞∂∆∇]', text)
        symbol_placeholders = {sym: f'__MATH_{i}__' for i, sym in enumerate(math_symbols)}
        
        for sym, placeholder in symbol_placeholders.items():
            text = text.replace(sym, placeholder)
        
        # Apply unidecode while preserving certain patterns
        text = unidecode.unidecode(text)
        
        # Restore mathematical symbols
        for sym, placeholder in symbol_placeholders.items():
            text = text.replace(placeholder, f' {sym} ')
        
        # Use Gemini for final formatting
        return self.format_text_with_gemini(text)

    def format_markdown_text(self, text: str) -> str:
        """
        Format text with proper markdown and LaTeX syntax.
        
        Args:
            text (str): Text to format
            
        Returns:
            str: Formatted text
        """
        # Format mathematical expressions
        text = text.replace('||', '\\|\\|')  # Escape pipes for markdown
        
        # Format subscripts and superscripts
        import re
        text = re.sub(r'([a-zA-Z])(\d+)', r'\1_\2', text)  # Convert a1 to a_1
        
        # Handle bold text
        text = text.replace('**', '__')  # Convert ** to __ for better markdown rendering
        
        return text

def main():
    # Set page config
    st.set_page_config(
        page_title="Research Paper Q&A",
        page_icon="🔍",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This is a RAG-based Q&A system for research papers.
    
    **How to use:**
    1. Type your question in the text input
    2. Click 'Ask' or press Enter
    3. View the AI-generated answer
    4. Optionally expand to see the source context
    
    **Technologies used:**
    - Sentence Transformers
    - FAISS vector store
    - Gemini 2.0 Flash API
    - Streamlit
    """)
    
    # Main content
    st.title("Research Paper Q&A System")
    
    # Initialize QA system
    @st.cache_resource
    def get_qa_system():
        qa_system = RAGQASystem()
        if not qa_system.load_index():
            st.error("Error: No index found. Please build the index first.")
            return None
        return qa_system
    
    qa_system = get_qa_system()
    
    if qa_system:
        # Question input
        query = st.text_input(
            "Ask a question about the research papers:",
            placeholder="e.g., What are the main findings regarding..."
        )
        
        if query:
            with st.spinner("Searching for relevant information..."):
                results = qa_system.retrieve_context(query)
                if results:
                    # Clean and prepare context
                    cleaned_results = [(qa_system.clean_text(chunk), score) for chunk, score in results]
                    context = "\n\n".join([chunk for chunk, _ in cleaned_results])
                    
                    # Generate answer
                    answer = qa_system.generate_answer(query, context)
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    # Display context
                    with st.expander("View Source Context 📚"):
                        st.markdown("""
                        <style>
                        .context-box {
                            background-color: #f5f5f5;
                            border-left: 3px solid #2e7d32;
                            padding: 20px;
                            margin: 10px 0;
                            border-radius: 5px;
                            font-family: 'Times New Roman', serif;
                            line-height: 1.8;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                        }
                        .context-box p {
                            margin: 0;
                            padding: 0;
                            text-align: justify;
                            font-size: 16px;
                        }
                        .similarity-score {
                            color: #1976d2;
                            font-size: 0.9em;
                            font-weight: 500;
                            margin-bottom: 15px;
                            font-family: 'Arial', sans-serif;
                        }
                        .math {
                            font-family: 'Computer Modern', serif;
                        }
                        /* MathML styles */
                        math {
                            font-size: 1.1em;
                            font-family: 'Computer Modern', serif;
                        }
                        </style>
                        
                        <!-- MathJax Configuration -->
                        <script type="text/javascript">
                        window.MathJax = {
                            tex: {
                                inlineMath: [['$', '$']],
                                displayMath: [['\\[', '\\]']],
                                processEscapes: true
                            },
                            svg: {
                                fontCache: 'global'
                            }
                        };
                        </script>
                        <script type="text/javascript" id="MathJax-script" async
                            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
                        </script>
                        """, unsafe_allow_html=True)
                        
                        for i, (chunk, score) in enumerate(cleaned_results, 1):
                            rendered_chunk = qa_system.render_markdown_and_latex(chunk)
                            st.markdown(f"""
                            <div class='context-box'>
                                <p class='similarity-score'>📍 Source {i} • Relevance Score: {score:.4f}</p>
                                {rendered_chunk}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("No relevant information found.")

if __name__ == "__main__":
    main()
