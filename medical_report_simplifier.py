import os
import re
import argparse
import torch
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- 1. Configure Gemini API ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
vvvvvvvvvvvvvv
api_key = os.getenv("GEMINI_API_KEY")
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

# --- 2. Initialize Models (Done globally so they aren't reloaded every time) ---
print("Loading BioBERT model and tokenizer...")
biobert_model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
biobert_model = AutoModel.from_pretrained(biobert_model_name)

print("Loading NER Pipeline...")
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")

# --- 3. Core Functions ---
def extract_text(file_path):
    """Extracts text from a given PDF or Image file using Tesseract OCR."""
    print("Extracting text from document...")
    text = ""
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or image file (.png, .jpg, .jpeg, etc.).")
    return text

def extract_and_embed_medical_terms(text):
    """Extracts medical terms using NER and generates their BioBERT embeddings."""
    print("Extracting medical terms from text...")
    entities = ner_pipeline(text)
    
    # Extract unique terms longer than 2 characters
    terms = list(set([ent['word'] for ent in entities if len(ent['word']) > 2]))
    print(f"Found {len(terms)} unique medical terms.")
    
    term_embeddings = {}
    print("Generating BioBERT embeddings for terms...")
    for term in terms:
        inputs = tokenizer(term, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = biobert_model(**inputs)
            # Use the CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            term_embeddings[term] = embedding
            
    return terms, term_embeddings

def simplify_medical_report(raw_text, medical_terms):
    """Uses the LLM to simplify the raw OCR text and explain key medical terms."""
    if not api_key:
        return "Error: API key not configured. Cannot call LLM."

    prompt = f"""
You are a helpful, empathetic medical assistant. Your job is to simplify a medical report for a patient who has no medical background.

Here is the raw text extracted from the patient's medical report:
\"\"\"
{raw_text}
\"\"\"

Key medical terms we identified in the report:
{', '.join(medical_terms) if medical_terms else 'None specifically identified'}

Please provide a response with the following sections:
1. **Summary**: A brief, easy-to-understand summary of the report.
2. **Key Findings (What are the problems?)**: Explain the abnormal results or main findings in simple terms. Avoid complex jargon.
3. **Explanation of Medical Terms**: Briefly define the key medical terms found in the report. (Use the list above to guide you, but prioritize terms that are crucial for understanding the report).
4. **General Advice**: Provide general, non-diagnostic advice (e.g., "Consult your doctor for a detailed diagnosis", "Stay hydrated", etc.).

IMPORTANT: Maintain a supportive and objective tone. Add a disclaimer that you are an AI assistant and this is not a substitute for professional medical advice.
"""
    
    print("Sending request to LLM to generate a simplified report...")
    response = model.generate_content(prompt)
    return response.text

def process_medical_report(file_path):
    """Main pipeline to process a single medical report."""
    print(f"\n--- Processing {file_path} ---")
    try:
        # 1. OCR
        raw_text = extract_text(file_path)
        print("✅ Text successfully extracted.")
        
        # 2. Term Extraction & Embedding
        terms, embeddings = extract_and_embed_medical_terms(raw_text)
        print(f"✅ Extracted {len(terms)} medical terms and generated their embeddings.")
        
        # 3. LLM Simplification
        simplified_output = simplify_medical_report(raw_text, terms)
        
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
