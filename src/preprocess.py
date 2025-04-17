import pdfplumber
import re
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import os
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

nltk.download('stopwords')
nltk.download('punkt')

def extract_clean_text(pdf_path: str) -> str:
    """
    Extract clean text from a PDF file, removing headers, footers, page numbers,
    and the References section.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Cleaned text content from the PDF
        
    Raises:
        Exception: If there's an error processing the PDF file
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Store all extracted text
            all_text = []
            
            for page in pdf.pages:
                try:
                    # Extract text from the entire page if dimensions are invalid
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                except Exception as e:
                    logger.warning(f"Error extracting page from {pdf_path}: {str(e)}")
                    # Try to extract text from entire page without cropping
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
            
            if not all_text:
                raise Exception("No text could be extracted from any page")
            
            # Join all text
            full_text = '\n'.join(all_text)
            
            # Remove page numbers (various formats)
            full_text = re.sub(r'\b\d+\b(?=\s*$)', '', full_text, flags=re.MULTILINE)
            
            # Remove References section if it exists
            # Common variations of reference section headers
            ref_patterns = [
                r'References\s*\n',
                r'REFERENCES\s*\n',
                r'Bibliography\s*\n',
                r'BIBLIOGRAPHY\s*\n',
                r'Works Cited\s*\n',
                r'References Cited\s*\n'
            ]
            
            # Find the earliest occurrence of any reference section
            ref_positions = []
            for pattern in ref_patterns:
                match = re.search(pattern, full_text)
                if match:
                    ref_positions.append(match.start())
            
            # If references section was found, truncate text
            if ref_positions:
                full_text = full_text[:min(ref_positions)]
            
            # Clean up extra whitespace
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            
            return clean_text
            
    except Exception as e:
        raise Exception(f"Error processing PDF file: {str(e)}")

def is_likely_header_or_footer(text: str) -> bool:
    """
    Check if a line of text is likely to be a header or footer.
    
    Args:
        text (str): Line of text to check
        
    Returns:
        bool: True if the text is likely a header/footer, False otherwise
    """
    # Common patterns for headers and footers
    patterns = [
        r'^\d+$',  # Page numbers
        r'^Page \d+$',  # "Page X" format
        r'^\s*\d+\s*$',  # Page numbers with whitespace
        r'^Chapter \d+',  # Chapter headers
        r'^\s*[A-Za-z\s]+\s*\|\s*\d+\s*$',  # Journal style headers
        r'^\s*[A-Za-z\s]+\s*\d{4}\s*$',  # Journal name with year
    ]
    
    return any(re.match(pattern, text.strip()) for pattern in patterns)

def split_into_sections(text: str) -> List[str]:
    """
    Split the text into sections based on common section headers.
    
    Args:
        text (str): Input text to split
        
    Returns:
        List[str]: List of text sections
    """
    # Common section headers in research papers
    section_patterns = [
        r'Abstract',
        r'Introduction',
        r'Background',
        r'Methods?',
        r'Methodology',
        r'Results?',
        r'Discussion',
        r'Conclusion',
        r'References'
    ]
    
    # Create regex pattern for section headers
    pattern = r'(?i)^\s*(' + '|'.join(section_patterns) + r')\s*$'
    
    # Split text into sections
    sections = re.split(pattern, text, flags=re.MULTILINE)
    
    # Remove empty sections and strip whitespace
    sections = [s.strip() for s in sections if s.strip()]
    
    return sections

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocess text by converting to lowercase, removing special characters,
    numbers, and extra whitespace. Optionally remove stopwords.
    
    Args:
        text (str): Input text to clean
        remove_stopwords (bool): Whether to remove stopwords (default: False)
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    
    return text

def process_earc_dataset(base_path: str = "data/raw/EARC_Dataset/Reference") -> pd.DataFrame:
    """
    Process the EARC dataset by reading PDFs from Publishable and Non-Publishable folders,
    extracting clean text, and creating a labeled DataFrame.
    
    Args:
        base_path (str): Base path to the EARC dataset Reference folder
        
    Returns:
        pd.DataFrame: DataFrame containing Paper_ID, Cleaned_Text, and Label
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize lists to store data
    data = []
    
    # Process both Publishable and Non-Publishable folders
    for label in ['Publishable', 'Non-Publishable']:
        folder_path = os.path.join(base_path, label)
        
        # Check if directory exists
        if not os.path.exists(folder_path):
            logger.error(f"Directory not found: {folder_path}")
            continue
            
        logger.info(f"Processing {label} papers...")
        
        # Process each PDF in the folder
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        for filename in tqdm(pdf_files):
            try:
                # Extract Paper ID from filename (assuming filename is the ID)
                paper_id = os.path.splitext(filename)[0]
                
                # Get full path to PDF
                pdf_path = os.path.join(folder_path, filename)
                
                # Extract and clean text
                raw_text = extract_clean_text(pdf_path)
                if not raw_text:
                    logger.warning(f"No text extracted from {filename}")
                    continue
                
                # Store the data
                data.append({
                    'Paper_ID': paper_id,
                    'Cleaned_Text': raw_text,
                    'Label': label
                })
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Log statistics only if DataFrame is not empty
    if not df.empty:
        logger.info(f"Total papers processed: {len(df)}")
        logger.info(f"Publishable papers: {len(df[df['Label'] == 'Publishable'])}")
        logger.info(f"Non-Publishable papers: {len(df[df['Label'] == 'Non-Publishable'])}")
    else:
        logger.warning("No papers were successfully processed")
        # Initialize empty DataFrame with correct columns
        df = pd.DataFrame(columns=['Paper_ID', 'Cleaned_Text', 'Label'])
    
    return df

def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed"):
    """
    Save the processed DataFrame to CSV and optionally pickle format.
    
    Args:
        df (pd.DataFrame): Processed DataFrame to save
        output_path (str): Directory to save the processed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_path, 'processed_papers.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as pickle for preserving data types
    pickle_path = os.path.join(output_path, 'processed_papers.pkl')
    df.to_pickle(pickle_path)
    
    print(f"Data saved to {csv_path} and {pickle_path}")

def preprocess_paper(file_path: str, min_word_count: int = 50) -> Optional[str]:
    """
    Extract and preprocess text from a PDF paper.
    
    Args:
        file_path (str): Path to the PDF file
        min_word_count (int): Minimum number of words required in processed text (default: 50)
        
    Returns:
        Optional[str]: Preprocessed text or None if extraction fails
    """
    try:
        # Extract text from PDF
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return None
            
        # Remove References section if it exists
        ref_patterns = [
            r'References\s*\n',
            r'REFERENCES\s*\n',
            r'Bibliography\s*\n',
            r'BIBLIOGRAPHY\s*\n',
            r'Works Cited\s*\n',
            r'References Cited\s*\n'
        ]
        
        # Find the earliest occurrence of any reference section
        ref_positions = []
        for pattern in ref_patterns:
            match = re.search(pattern, text)
            if match:
                ref_positions.append(match.start())
                
        # If references section was found, truncate text
        if ref_positions:
            text = text[:min(ref_positions)]
        
        # Basic preprocessing
        text = text.lower()  # Convert to lowercase
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        filtered_lines = [
            line for line in lines 
            if len(line.strip()) > 20  # Remove short lines (likely headers/footers)
            and not re.match(r'^\d+$', line.strip())  # Remove page numbers
        ]
        text = ' '.join(filtered_lines)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Join words back together
        processed_text = ' '.join(filtered_words)
        
        # Verify we still have meaningful text
        if len(processed_text.split()) < min_word_count:
            logger.warning(f"Processed text too short for {file_path}")
            return None
            
        return processed_text
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Process the dataset
    df = process_earc_dataset()
    
    # Save the processed data
    save_processed_data(df)
