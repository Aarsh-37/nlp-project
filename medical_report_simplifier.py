import os
import re
import argparse
import time
import requests
import torch
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import pipeline
from pdf2image.exceptions import PDFInfoNotInstalledError
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
import streamlit as st

# Load environment variables
load_dotenv()

# --- 1. Configure Supported AI LLM APIs and OCR ---
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

ocr_space_api_key = os.getenv("OCR_SPACE_API_KEY")

# Initialize Gemini
gemini_model = None
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.3, "top_p": 0.95, "top_k": 40,
            "max_output_tokens": 8192, "response_mime_type": "application/json",
        }
    )

# Initialize Grok
grok_client = None
if os.getenv("GROK_API_KEY"):
    grok_client = OpenAI(api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1")

# Initialize DeepSeek
deepseek_client = None
if os.getenv("DEEPSEEK_API_KEY"):
    deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


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
    valid_chunks = [c for c in chunks if c.strip()]
    
    if valid_chunks:
        try:
            # Batch process all chunks at once to minimize python overhead
            batch_entities = ner_pipeline(valid_chunks)
            
            # HuggingFace sometimes returns a flat list if only 1 string is passed
            if len(valid_chunks) == 1 and (len(batch_entities) == 0 or isinstance(batch_entities[0], dict)):
                batch_entities = [batch_entities]
                
            for entities in batch_entities:
                for ent in entities:
                    if isinstance(ent, dict) and len(ent.get('word', '')) > 2:
                        terms.add(ent['word'])
        except Exception as e:
            print(f"Warning: NER failed during batch processing: {e}")
            
    terms_list = list(terms)
    print(f"Found {len(terms_list)} unique medical terms.")
    return terms_list

def simplify_medical_report(raw_text, medical_terms, llm_choice="Gemini 2.5 Flash"):
    """Uses the requested LLM to generate BOTH the brief summary and detailed explanation in ONE single fast JSON API call."""
    
    # --- PRIMARY RAG DICTIONARY LOOKUP ---
    # Load our simplified dictionary definitions to feed directly into the AI prompt
    import json
    LOCAL_MED_DICT = {}
    try:
        with open("data/medical_dict.json", "r", encoding="utf-8") as f:
            LOCAL_MED_DICT = json.load(f).get("medical_entities", {})
    except Exception as e:
        print(f"Warning: Could not load local dictionary for RAG: {e}")

    enriched_terms = []
    if medical_terms:
        for term in medical_terms:
            clean_term = term.lower().strip()
            definition = LOCAL_MED_DICT.get(clean_term)
            if definition:
                enriched_terms.append(f"'{term}' (Simplified Definition: {definition})")
            else:
                enriched_terms.append(f"'{term}'")

    prompt = f"""
You are a helpful, empathetic medical assistant. Your job is to simplify a medical report for a patient who has no medical background.

Here is the raw text extracted from the patient's medical report:
\"\"\"
{raw_text}
\"\"\"

Key medical terms we identified in the report (along with their strict dictionary definitions):
{', '.join(enriched_terms) if enriched_terms else 'None specifically identified'}

Please provide a response as a JSON object with EXACTLY the following schema:
{{
  "brief_summary": "A very brief, and highly simplified summary (3-4 sentences maximum). Focus only on the most important takeaways and overall health status, tailored for a patient with no medical background.",
  "detailed_report": "A detailed explanation structured naturally with markdown format containing sections like 'Summary', 'Key Findings (What are the problems?)', 'Explanation of Medical Terms' (Crucial: You MUST use the strict 'Simplified Definitions' provided above when explaining these terms), and 'General Advice' (max 200-300 words). Maintain a supportive tone."
}}

IMPORTANT: Add a disclaimer at the end of the detailed_report that you are an AI assistant and this is not a substitute for professional medical advice.
"""
    
    print(f"Sending batched request to {llm_choice} LLM to generate both summary and detailed report simultaneously...")
    try:
        import json
        if llm_choice == "Gemini 2.5 Flash":
            if not gemini_model:
                raise Exception("GEMINI_API_KEY not found in .env")
            response = gemini_model.generate_content(prompt)
            result = json.loads(response.text)
            return result
            
        elif llm_choice == "Grok (xAI)":
            if not grok_client:
                raise Exception("GROK_API_KEY not found in .env")
            response = grok_client.chat.completions.create(
                model="grok-beta", 
                messages=[
                    {"role": "system", "content": "You are a helpful, empathetic medical assistant. Please respond ONLY with the requested JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            return result
            
        elif llm_choice == "DeepSeek API":
            if not deepseek_client:
                raise Exception("DEEPSEEK_API_KEY not found in .env")
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": "You are a helpful, empathetic medical assistant. Please respond ONLY with the requested JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            return result
            
    except Exception as e:
        print(f"LLM Processing Failed ({llm_choice}): {e}")
        # --- RULE-BASED FALLBACK SYSTEM ---
        # If the API key is invalid, quota exceeded, or internet drops, we rescue the app here.
        # We perform local RAG by pulling definitions directly from our JSON dictionary!
        import json
        LOCAL_MED_DICT = {}
        try:
            with open("data/medical_dict.json", "r", encoding="utf-8") as f:
                LOCAL_MED_DICT = json.load(f).get("medical_entities", {})
        except Exception as file_e:
            print(f"Warning: Could not load local dictionary for fallback: {file_e}")

        fallback_summary = "⚠️ The AI Simplifier is currently unavailable (Network/API Error). Below is the simplified data extracted directly from your document's terminology."
        
        fallback_report = "## ⚙️ Automated Outline (Offline Mode)\nOur clinical AI connection failed, but our local pipeline successfully identified the following key medical terms in your document:\n\n"
        if medical_terms:
            for term in medical_terms:
                clean_term = term.lower().strip()
                definition = LOCAL_MED_DICT.get(clean_term)
                if definition:
                    fallback_report += f"- **{term.title()}**: {definition}\n"
                else:
                    fallback_report += f"- **{term.title()}**\n"
        else:
            fallback_report += "*No specific medical terminology identified.*"
            
        fallback_report += "\n\n*(Raw OCR text was successfully processed, but detailed AI analysis is offline. Please check your API key or internet connection.)*"
        
        return {
            "brief_summary": fallback_summary,
            "detailed_report": fallback_report
        }

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
        result_dict = simplify_medical_report(raw_text, terms)
        llm_time = time.time() - start_llm
        print(f"✅ LLM generated summary & report in {llm_time:.2f}s.")
        
        total_time = time.time() - start_total
        
        print("\n==================================================")
        print("🩺 SIMPLIFIED MEDICAL REPORT")
        print("==================================================")
        print(result_dict.get("detailed_report", "Error."))
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
