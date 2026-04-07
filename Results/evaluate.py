import json
import warnings
import jiwer
from rouge_score import rouge_scorer
from bert_score import score

warnings.filterwarnings("ignore")

def evaluate_ocr(reference_text, predicted_text):
    """Calculates Word Error Rate (WER) using jiwer."""
    if not reference_text.strip(): return 0.0
    ref = " ".join(reference_text.strip().split())
    pred = " ".join(predicted_text.strip().split())
    wer = jiwer.wer(ref, pred)
    return wer

def evaluate_ner(reference_terms, predicted_terms):
    """Calculates Token-level/Set-based Precision, Recall, and F1-score for NER."""
    ref_set = set([t.lower() for t in reference_terms])
    pred_set = set([t.lower() for t in predicted_terms])
    
    tp = len(ref_set.intersection(pred_set))
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_llm_generation(reference_summary, predicted_summary):
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

if __name__ == "__main__":
    print("=" * 50)
    print("   EVALUATION METRICS   ")
    print("=" * 50)

    # 1. OCR Accuracy
    ref_ocr = "Patient has a history of severe hypertension and type 2 diabetes mellitus. Blood pressure is 160/95."
    pred_ocr = "Patient has a histry of severe hypertension and ty pe 2 diabetss mellitus. Blood presure is 160/95 ."
    wer_score = evaluate_ocr(ref_ocr, pred_ocr)
    print("\n[1] OCR Accuracy")
    print(f"    Word Error Rate (WER): {wer_score:.4f}")

    # 2. Medical Term Detection (NER Performance)
    # Using the subset matched using d4data/biomedical-ner-all against medical_dict.json
    ref_ner = ["hypertension", "tachycardia", "glucose", "diabetes mellitus", "hemoglobin", "anemia", "ecg", "mri", "myocardial infarction"]
    pred_ner = ["hypertension", "tachycardia", "glucose", "diabetes", "hemoglobin", "anemia", "ecg", "mri", "myocardial", "infarction"]
    ner_scores = evaluate_ner(ref_ner, pred_ner)
    print("\n[2] Medical Term Detection (NER Performance)")
    print(f"    Precision: {ner_scores['precision']:.4f}")
    print(f"    Recall:    {ner_scores['recall']:.4f}")
    print(f"    F1-score:  {ner_scores['f1']:.4f}")

    # 3. LLM Generation Quality
    ref_summary = "The patient suffers from prolonged high blood pressure and type 2 diabetes."
    pred_summary = "The patient has severe high blood pressure and type 2 diabetes."
    print("\n[3] LLM Generation Quality (Rule-Based)")
    print("    Calculating ROUGE & BERTScore (please wait)...")
    try:
        llm_metrics = evaluate_llm_generation(ref_summary, pred_summary)
        print("\n    ROUGE Score (Text Overlap):")
        print(f"      - ROUGE-1: {llm_metrics['rouge1_fmeasure']:.4f}")
        print(f"      - ROUGE-2: {llm_metrics['rouge2_fmeasure']:.4f}")
        print(f"      - ROUGE-L: {llm_metrics['rougeL_fmeasure']:.4f}")
        print("\n    BERTScore (Semantic Similarity):")
        print(f"      - BERTScore F1: {llm_metrics['bertscore_f1']:.4f}")
    except Exception as e:
        print(f"    Error calculating LLM metrics: {e}")

    print("\n" + "=" * 50)
