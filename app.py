import os
import tempfile
import time
import re
import streamlit as st
from PIL import Image
from fpdf import FPDF

# Import the logic from our script
# Note: we are importing functions, not running the CLI block
from medical_report_simplifier import extract_text, extract_medical_terms, simplify_medical_report, get_brief_summary

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Report Simplifier",
    page_icon="🩺",
    layout="centered"
)

def create_pdf(summary, detailed_report):
    """Generates a PDF byte string from the simplified report text."""
    # FPDF only natively supports latin-1. Emojis and smart quotes cause crashes.
    # So we strip out non-ascii characters for the PDF locally.
    clean_summary = re.sub(r'[^\x00-\x7F]+', ' ', summary)
    clean_report = re.sub(r'[^\x00-\x7F]+', ' ', detailed_report)
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Simplified Medical Report", ln=True, align="C")
    pdf.ln(10)
    
    # Brief Summary Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Brief Summary:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, txt=clean_summary)
    pdf.ln(10)
    
    # Detailed Report Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Detailed Report:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, txt=clean_report)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, txt="Disclaimer: This report was simplified by AI and is not valid medical advice.", ln=True)
    
    # Save to a temp string and return bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.remove(tmp.name)
    
    return pdf_bytes

st.title("🩺 Medical Report Simplifier")
st.markdown("""
Welcome! Upload your medical report (PDF or Image) below. 
Our AI will extract the text, identify key medical terms, and provide a simplified, patient-friendly explanation.

*Disclaimer: This is an AI assistant, not a doctor. Always consult a healthcare professional for medical advice.*
""")

# File uploader
uploaded_file = st.file_uploader("Upload your medical report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image (if it's an image)
    if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        st.image(uploaded_file, caption="Uploaded Report", use_container_width=True)
    
    st.info("File uploaded successfully. Click the button below to analyze it.")
    
    if st.button("Analyze & Simplify Report"):
        with st.spinner("Processing your report... This may take a minute."):
            try:
                # Save the uploaded file temporarily so our existing functions can process it via file_path
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                start_total = time.time()
                
                # Step 1: OCR
                st.text("📝 Extracting text via OCR...")
                start_ocr = time.time()
                raw_text = extract_text(tmp_file_path)
                ocr_time = time.time() - start_ocr
                
                if not raw_text or len(raw_text.strip()) < 10:
                    st.warning("⚠️ Could not extract enough readable text from this file. Please ensure the image/PDF is clear and contains text.")
                    os.remove(tmp_file_path)
                    st.stop()
                    
                # Step 2: Extract Terms
                st.text("🧬 Finding medical terminology...")
                start_ner = time.time()
                terms = extract_medical_terms(raw_text)
                ner_time = time.time() - start_ner
                
                # Step 3: LLM Simplification
                st.text("Generating simplified explanation...")
                start_llm = time.time()
                simplified_output = simplify_medical_report(raw_text, terms)
                llm_time = time.time() - start_llm
                
                # Step 4: Brief Summary
                st.text("Generating brief summary...")
                start_summary = time.time()
                brief_summary = get_brief_summary(simplified_output)
                summary_time = time.time() - start_summary
                
                total_time = time.time() - start_total
                
                # Clean up temporary file
                os.remove(tmp_file_path)
                
                # Display Results
                st.success(f"Analysis Complete in {total_time:.2f} seconds!")
                st.markdown("---")
                
                tab1, tab2 ,tab3= st.tabs(["📝 Brief Summary", "🧾 Detailed Simplified Report","⏱️ Pipeline Performance"])
                
                with tab1:
                    st.subheader("📝 Brief Summary")
                    st.info(brief_summary)
                    
                with tab2:
                    st.subheader("🧾 Detailed Simplified Report")
                    st.markdown(simplified_output)
                    
                with tab3:
                    st.subheader("⏱️ Pipeline Performance")
                    col1, col2 = st.columns(2)
                    col1.metric("1. OCR Processing", f"{ocr_time:.2f}s")
                    col2.metric("2. NER Extraction", f"{ner_time:.2f}s")
                    
                    col3, col4 = st.columns(2)
                    col3.metric("3. LLM Detailed Report", f"{llm_time:.2f}s")
                    col4.metric("4. LLM Brief Summary", f"{summary_time:.2f}s")
                    
                    st.divider()
                    st.metric("Total Execution Time", f"{total_time:.2f}s", help="Includes all processing steps.")
                
                with st.expander("View Raw Extracted Text (For Reference)"):
                    st.text(raw_text)
                    
                st.markdown("---")
                # Generate and present the PDF download button
                pdf_bytes = create_pdf(brief_summary, simplified_output)
                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf_bytes,
                    file_name="simplified_medical_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
