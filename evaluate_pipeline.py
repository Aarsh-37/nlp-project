import jiwer
import textstat
from rouge_score import rouge_scorer
from bert_score import score

def calculate_ocr_wer(reference_text, predicted_text):
    """Calculates Word Error Rate (WER) using jiwer."""
    if not reference_text.strip(): return 0.0
    # Clean texts
    ref = " ".join(reference_text.strip().split())
    pred = " ".join(predicted_text.strip().split())
    wer = jiwer.wer(ref, pred)
    return wer

def calculate_ner_f1(reference_terms, predicted_terms):
    """Calculates Token-level/Set-based F1-score for NER."""
    ref_set = set([t.lower() for t in reference_terms])
    pred_set = set([t.lower() for t in predicted_terms])
    
    true_positives = len(ref_set.intersection(pred_set))
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_llm_metrics(reference_summary, predicted_summary):
    """Calculates ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore for text generation."""
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_summary, predicted_summary)
    
    # BERTScore
    P, R, F1 = score([predicted_summary], [reference_summary], lang="en", verbose=False)
    
    return {
        "rouge1_fmeasure": rouge_scores['rouge1'].fmeasure,
        "rouge2_fmeasure": rouge_scores['rouge2'].fmeasure,
        "rougeL_fmeasure": rouge_scores['rougeL'].fmeasure,
        "bertscore_f1": F1.mean().item()
    }

def calculate_ux_readability(text):
    """Calculates Flesch Reading Ease and Flesch-Kincaid Grade Level."""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
    }

def print_human_eval_template():
    """Prints a template for Human Evaluation."""
    template = """
==================================================
🧑‍⚕️ HUMAN EVALUATION TEMPLATE 
==================================================
Please rate the final output on a scale of 1 to 5 (1 = Poor, 5 = Excellent)

1. MEDICAL ACCURACY (1-5): ____
   - Did the AI hallucinate or drop important details?

2. READABILITY & CLARITY (1-5): ____
   - Was the explanation easy for a non-medical person to understand?

3. OVERALL USEFULNESS (1-5): ____
   - Would a patient find this summary helpful?

Notes/Comments: _____________________________________
==================================================
"""
    print(template)


if __name__ == "__main__":
    print("Running DUMMY EVALUATION DATASET...")
    print("-----------------------------------")
    
    # 1. OCR Dummy Data (Assume typos during Tesseract parsing)
    ref_ocr = "Patient has a history of severe hypertension and type 2 diabetes mellitus. Blood pressure is 160/95."
    pred_ocr = "Patient has a histry of severe hypertension and ty pe 2 diabetss mellitus. Blood presure is 160/95 ."
    
    wer_score = calculate_ocr_wer(ref_ocr, pred_ocr)
    print(f"👉 OCR WER (Word Error Rate): {wer_score:.2f} (Lower = Better)")
    
    # 2. NER Dummy Data (Simulated Missing and hallucinated entities)
    ref_ner = ["hypertension", "diabetes mellitus", "blood pressure"]
    pred_ner = ["hypertension", "diabetss mellitus", "blood", "pressure"]
    
    ner_scores = calculate_ner_f1(ref_ner, pred_ner)
    print(f"👉 NER F1-score: {ner_scores['f1']:.2f} (Precision: {ner_scores['precision']:.2f}, Recall: {ner_scores['recall']:.2f})")
    
    # 3. LLM Pipeline Dummy Data
    ref_summary = "The patient suffers from prolonged high blood pressure and type 2 diabetes."
    pred_summary = "The patient has severe high blood pressure and type 2 diabetes."
    
    print("\nCalculating LLM Generation Metrics (ROUGE & BERTScore)...")
    print("Please wait, loading BERTScore models can take ~30 seconds...")
    try:
        import warnings
        warnings.filterwarnings('ignore') # hide huggingface loading warnings
        llm_metrics = calculate_llm_metrics(ref_summary, pred_summary)
        print(f"👉 LLM ROUGE-1 F1: {llm_metrics['rouge1_fmeasure']:.2f}")
        print(f"👉 LLM ROUGE-2 F1: {llm_metrics['rouge2_fmeasure']:.2f}")
        print(f"👉 LLM ROUGE-L F1: {llm_metrics['rougeL_fmeasure']:.2f}")
        print(f"👉 LLM BERTScore : {llm_metrics['bertscore_f1']:.4f}")
    except Exception as e:
        print(f"Failed to calculate BERTScore/ROUGE. Error: {e}")
        
    # 4. UX Readability Data
    read_metrics = calculate_ux_readability(pred_summary)
    print("\n👉 UX Readability Metrics:")
    print(f"   - Flesch Reading Ease: {read_metrics['flesch_reading_ease']:.2f} (Higher = Better/Easier)")
    print(f"   - Flesch-Kincaid Grade: {read_metrics['flesch_kincaid_grade']:.1f} (Lower = Lower education level required)")
    
    # 5. Human Evaluation Form
    print_human_eval_template()
    
    print("\n✅ Evaluation Script Ready. To use this with your own dataset, import these functions and pass your arrays of reference vs. predicted text!")
