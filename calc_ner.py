import json
from transformers import pipeline

# 1. Load the dictionary
with open("data/medical_dict.json", "r") as f:
    med_dict = json.load(f)

vocab_terms = list(med_dict["medical_entities"].keys())

# 2. Let's create a realistic medical report using terms from the dictionary
reference_text = (
    "The patient presented with severe hypertension and tachycardia. "
    "Lab results show high glucose indicating diabetes mellitus, and reduced hemoglobin indicating anemia. "
    "We recommend an ecg and mri for further evaluation of suspected myocardial infarction."
)

# The ground-truth terms (from our dictionary) that appear in this text:
ref_terms = [
    "hypertension", "tachycardia", "glucose", "diabetes mellitus", 
    "hemoglobin", "anemia", "ecg", "mri", "myocardial infarction"
]

# 3. Predict using the actual pipeline
print("Loading NER Pipeline...")
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")

print("Extracting terms...")
entities = ner_pipeline(reference_text)
pred_terms = {ent['word'].lower() for ent in entities if len(ent['word']) > 2}

# 4. Calculate Precision/Recall/F1
ref_set = set(t.lower() for t in ref_terms)
pred_set = pred_terms

tp = len(ref_set & pred_set)
fp = len(pred_set - ref_set)
fn = len(ref_set - pred_set)

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1_ner    = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

print(f"\n--- NER EVALUATION ---")
print(f"Reference Terms: {sorted(ref_set)}")
print(f"Predicted Terms: {sorted(pred_set)}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_ner:.4f}")

