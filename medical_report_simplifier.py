import os
import re
import argparse
import time
import requests
import torch
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import pipeline
from pdf2image.exceptions import PDFInfoNotInstalledError
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()

# --- 1. Configure Gemini API  and OCR SPACE API KEY ---
if os.name == "nt":  # Windows (your local machine)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

api_key = os.getenv("GEMINI_API_KEY")
ocr_space_api_key = os.getenv("OCR_SPACE_API_KEY")

if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment or .env file.")
else:
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
    )
    print("✅ API Key loaded successfully.")


# --- 2. Initialize Models ---

@st.cache_resource(show_spinner="Loading Medical AI Model into RAM (this only happens once)...")
def get_ner_pipeline():
    """Loads and caches the heavy NER model so it isn't reloaded on every Streamlit run."""
    print("Loading NER Pipeline...")
    return pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")


# --- 3. Core Functions ---
def extract_text_ocr_space(file_path):
    """Sends file to OCR Space API and parses response."""
    if not ocr_space_api_key:
        raise ValueError("OCR_SPACE_API_KEY not set.")
    
    payload = {
        'apikey': ocr_space_api_key,
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': 2 # Engine 2 is often better for numbers/messy text
    }
    
    with open(file_path, 'rb') as f:
        r = requests.post(
            'https://api.ocr.space/parse/image',
            files={file_path: f},
            data=payload,
        )
    
    result = r.json()
    
    if result.get('IsErroredOnProcessing'):
        error_msg = result.get('ErrorMessage', ['Unknown API error'])[0]
        raise RuntimeError(f"OCR Space Error: {error_msg}")
        
    parsed_text = ""
    for result_dict in result.get('ParsedResults', []):
        parsed_text += result_dict.get('ParsedText', '') + "\n"
        
    return parsed_text

def extract_text(file_path):
    """Extracts text, trying OCR Space first, then falling back to Tesseract."""
    print("Extracting text from document...")
    
    # 1. Attempt OCR Space if configured
    if ocr_space_api_key:
        try:
            print("Attempting OCR using OCR Space API...")
            text = extract_text_ocr_space(file_path)
            if text.strip():
                print("✅ Successfully used OCR Space API.")
                return text
        except Exception as e:
            print(f"⚠️ OCR Space failed: {e}. Falling back to local Tesseract...")
            
    # 2. Fallback to Local Tesseract
    print("Running local Tesseract OCR...")
    text = ""
    if file_path.lower().endswith('.pdf'):
        try:
            pages = convert_from_path(file_path)
            for page in pages:
                text += pytesseract.image_to_string(page) + "\n"
        except PDFInfoNotInstalledError:
            raise RuntimeError("PDF processing requires Poppler. Please install Poppler and add its binary folder to your system PATH (Windows), or use an image instead.")
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or image file (.png, .jpg, .jpeg, etc.).")
    return text

def extract_medical_terms(text):
    """Extracts medical terms using NER with text chunking to avoid max length errors."""
    print("Extracting medical terms from text...")
    

    # Characters (safe sub-token limit for 512 length)
    chunk_size = 1500  
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Grab our cached model!
    ner_pipeline = get_ner_pipeline()
    
    terms = set()
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            entities = ner_pipeline(chunk)
            for ent in entities:
                if len(ent['word']) > 2:
                    terms.add(ent['word'])
        except Exception as e:
            print(f"Warning: NER skipped a chunk due to an error: {e}")
            
    terms_list = list(terms)
    print(f"Found {len(terms_list)} unique medical terms.")
    return terms_list

def simplify_medical_report(raw_text, medical_terms):
    """Uses the LLM to simplify the raw OCR text and explain key medical terms."""
    if not api_key:
        return "Error: API key not configured. Cannot call LLM."

    prompt = f"""
You are a helpful, empathetic medical assistant. Your job is to simplify a medical report for a patient who has no medical background.

First, determine if the provided text is actually a medical report, clinical note, or laboratory results document. If the text does NOT appear to be a medical document (e.g., if it's a recipe, receipt, random article, or non-medical text), you MUST immediately abort and return EXACTLY the following error string, and nothing else:
ERROR: NOT_A_MEDICAL_REPORT

If it IS a valid medical document, here is the raw text extracted from it:
\"\"\"
{raw_text}
\"\"\"

Key medical terms we identified in the report:
{', '.join(medical_terms) if medical_terms else 'None specifically identified'}

If it is a medical report, please provide a response with the following sections:
1. **Summary**: A brief, easy-to-understand summary of the report.
2. **Key Findings (What are the problems?)**: Explain the abnormal results or main findings in simple terms. Avoid complex jargon.
3. **Explanation of Medical Terms**: Briefly define the key medical terms found in the report. (Use the list above to guide you, but prioritize terms that are crucial for understanding the report).
4. **General Advice**: Provide general, non-diagnostic advice (e.g., "Consult your doctor for a detailed diagnosis", "Stay hydrated", etc.).

IMPORTANT: Maintain a supportive and objective tone. Add a disclaimer that you are an AI assistant and this is not a substitute for professional medical advice.
give the output in max 200-300 words"""
    
    print("Sending request to LLM to generate a simplified report...")
    response = model.generate_content(prompt)
    return response.text

def get_brief_summary(text):
    """Generates a very short, easy to understand summary of the report."""
    if not api_key:
        return "Error: API key not configured. Cannot call LLM."

    prompt = f"""
You are a helpful medical assistant. Please provide a very brief, and highly simplified summary (3-4 sentences maximum) of the following medical report findings.
Focus only on the most important takeaways and overall health status, tailored for a patient with no medical background:

\"\"\"
{text}
\"\"\"
"""
    
    print("Sending request to LLM to generate a brief summary...")
    response = model.generate_content(prompt)
    return response.text

def process_medical_report(file_path):
    """Main pipeline to process a single medical report."""
    print(f"\n--- Processing {file_path} ---")
    try:
        start_total = time.time()
        
        # 1. OCR
        start_ocr = time.time()
        raw_text = extract_text(file_path)
        ocr_time = time.time() - start_ocr
        print(f"✅ Text successfully extracted in {ocr_time:.2f}s.")
        
        # 2. Term Extraction
        start_ner = time.time()
        terms = extract_medical_terms(raw_text)
        ner_time = time.time() - start_ner
        print(f"✅ Extracted {len(terms)} medical terms in {ner_time:.2f}s.")
        
        # 3. LLM Simplification
        start_llm = time.time()
        simplified_output = simplify_medical_report(raw_text, terms)
        llm_time = time.time() - start_llm
        print(f"✅ LLM generated report in {llm_time:.2f}s.")
        
        if "ERROR: NOT_A_MEDICAL_REPORT" in simplified_output:
            print("\n❌ ERROR: The provided document does not appear to be a medical report.")
            return

        total_time = time.time() - start_total
        
        print("\n==================================================")
        print("🩺 SIMPLIFIED MEDICAL REPORT")
        print("==================================================")
        print(simplified_output)
        print("==================================================\n")
        

    except Exception as e:
        print(f"An error occurred: {e}")

# --- 4. CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Report Simplifier CLI")
    parser.add_argument("file_path", help="Path to the patient's medical report (PDF or Image)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.file_path):
        process_medical_report(args.file_path)
    else:
        print(f"Error: The file '{args.file_path}' does not exist.")
