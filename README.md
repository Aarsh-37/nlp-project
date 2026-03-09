# 🩺 Medical Report Simplifier (NLP Project)

## 📌 Overview

Medical reports often contain complex medical terminology that can be difficult for patients to understand.
The **Medical Report Simplifier** is an NLP-based system that extracts text from medical report images and converts complex medical jargon into simple, easy-to-understand explanations.

The system allows a patient to upload a medical report image and receive a simplified explanation of the findings along with general advice.

---

## 🎯 Objective

* Extract text from medical report images using OCR.
* Identify medical terms and test results.
* Convert complex medical terminology into simplified explanations.
* Provide general health advice based on the extracted information.

---

## ⚙️ How the System Works

1. **Upload Medical Report**

   * User uploads an image of a medical report.

2. **Text Extraction (OCR)**

   * Text is extracted using **Tesseract OCR**.

3. **Text Processing**

   * NLP techniques are used to detect medical terms and values.

4. **Medical Term Simplification**

   * Medical jargon is mapped to simple explanations using a predefined dataset/dictionary.

5. **Output**

   * The system generates a simplified explanation of the report.

---

## 🧠 Technologies Used

* **Python**
* **Natural Language Processing (NLP)**
* **Tesseract OCR**
* **spaCy**
* **Scikit-learn**
* **Transformers (optional)**
* **Jupyter Notebook**

---

## 📦 Libraries Used

* pytesseract
* pdf2image
* spacy
* scikit-learn
* transformers
* pillow
* matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
medical-report-simplifier/
│
├── dataset/
│   └── medical_terms_dataset.csv
│
├── images/
│   └── sample_reports/
│
├── notebooks/
│   └── medical_report_simplifier.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── utils/
│   └── preprocessing.py
│
├── requirements.txt
├── README.md
└── app.py
```

---

## 🚀 How to Run the Project

1️⃣ **Clone the repository**

```bash
git clone https://github.com/AryanDemi/NLP-Project-.4.git
cd NLP-Project-.4
```

2️⃣ **Install Tesseract OCR (Crucial)**
The text extraction relies on the Tesseract OCR engine.
* **Windows**: Download the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install it. Make sure the path matches `C:\Program Files\Tesseract-OCR\tesseract.exe` (or update it in `medical_report_simplifier.py`).
* **Mac/Linux**: Install via `brew install tesseract` or `sudo apt install tesseract-ocr`.

3️⃣ **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Set up API Keys**
Rename the `.env.example` file to `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

5️⃣ **Run the Application**

You have three ways to run this project:

**Option A: Web Interface (Recommended)**
```bash
python -m streamlit run app.py
```
This opens a beautiful frontend in your browser where you can upload your report directly.

**Option B: Standalone Script**
```bash
python medical_report_simplifier.py "path/to/your_report.pdf"
```

**Option C: Jupyter Notebook**
Open `Medical_Report_Simplifier.ipynb` in your preferred editor (VS Code, JupyterLab), change the file path in the last cell, and run all cells.

---

## 📊 Example Workflow

Input:

* Upload CBC report image

System Processing:

* Extracts text using OCR
* Detects terms like *Hemoglobin*, *WBC*, *Platelets*

Output:

* Simplified explanations such as:

Example:

* **Hemoglobin:** Protein in blood that carries oxygen.
* **Low Hemoglobin:** May indicate anemia.
* **Advice:** Consult a doctor if levels are significantly below normal.

---

## 🔍 Future Improvements

* Better **medical entity recognition**
* Integration with **medical knowledge graphs**
* Improved accuracy using **larger medical datasets**
* Multi-language support for diverse patient backgrounds

---

## ⚠️ Disclaimer

This system provides **general informational explanations only** and should **not replace professional medical advice**. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 👨‍💻 Author

Aryan Bokde
Aarsh Vaidya

Software Engineering / NLP Project
