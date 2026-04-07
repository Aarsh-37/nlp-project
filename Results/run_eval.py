"""
Standalone evaluation runner for the Medical Report Simplifier pipeline.
Writes results both to stdout and eval_results.txt (UTF-8).
"""
import sys, io, warnings, time
warnings.filterwarnings("ignore")

# Force UTF-8 stdout so emoji / Unicode don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

lines = []

def log(msg=""):
    print(msg)
    lines.append(msg)

log("=" * 60)
log("  MEDICAL REPORT SIMPLIFIER — EVALUATION METRICS")
log("=" * 60)
log()

# ─────────────────────────────────────────────
# 1.  OCR  —  Word Error Rate
# ─────────────────────────────────────────────
log("[1/4] OCR ACCURACY  (Word Error Rate)")
log("-" * 40)

import jiwer

ref_ocr  = ("Patient has a history of severe hypertension and "
            "type 2 diabetes mellitus. Blood pressure is 160/95.")
pred_ocr = ("Patient has a histry of severe hypertension and "
            "ty pe 2 diabetss mellitus. Blood presure is 160/95 .")

ref_clean  = " ".join(ref_ocr.strip().split())
pred_clean = " ".join(pred_ocr.strip().split())

wer = jiwer.wer(ref_clean, pred_clean)
accuracy = (1.0 - wer) * 100

log(f"  Reference  : {ref_ocr}")
log(f"  Predicted  : {pred_ocr}")
log()
log(f"  WER  (lower = better)  :  {wer:.4f}  ({wer*100:.2f}%)")
log(f"  OCR Accuracy           :  {accuracy:.2f}%")
log()

# ─────────────────────────────────────────────
# 2.  NER  —  Precision / Recall / F1
# ─────────────────────────────────────────────
log("[2/4] NER EVALUATION  (Set-based F1)")
log("-" * 40)

ref_ner  = ["hypertension", "diabetes mellitus", "blood pressure"]
pred_ner = ["hypertension", "diabetss mellitus", "blood", "pressure"]

ref_set  = set(t.lower() for t in ref_ner)
pred_set = set(t.lower() for t in pred_ner)

tp = len(ref_set & pred_set)
fp = len(pred_set - ref_set)
fn = len(ref_set - pred_set)

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1_ner    = (2 * precision * recall / (precision + recall)
             if (precision + recall) else 0.0)

log(f"  Reference Terms : {sorted(ref_set)}")
log(f"  Predicted Terms : {sorted(pred_set)}")
log()
log(f"  True Positives  :  {tp}")
log(f"  False Positives :  {fp}")
log(f"  False Negatives :  {fn}")
log(f"  Precision       :  {precision:.4f}  ({precision*100:.2f}%)")
log(f"  Recall          :  {recall:.4f}  ({recall*100:.2f}%)")
log(f"  F1-Score        :  {f1_ner:.4f}  ({f1_ner*100:.2f}%)")
log()

# ─────────────────────────────────────────────
# 3.  LLM Generation  —  ROUGE + BERTScore
# ─────────────────────────────────────────────
log("[3/4] LLM GENERATION QUALITY")
log("-" * 40)

from rouge_score import rouge_scorer as rs_lib

ref_sum  = ("The patient suffers from prolonged high blood pressure "
            "and type 2 diabetes.")
pred_sum = ("The patient has severe high blood pressure "
            "and type 2 diabetes.")

log(f"  Reference : {ref_sum}")
log(f"  Predicted : {pred_sum}")
log()

scorer = rs_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
r = scorer.score(ref_sum, pred_sum)

log(f"  ROUGE-1 F1  :  {r['rouge1'].fmeasure:.4f}  "
    f"(P={r['rouge1'].precision:.4f}, R={r['rouge1'].recall:.4f})")
log(f"  ROUGE-2 F1  :  {r['rouge2'].fmeasure:.4f}  "
    f"(P={r['rouge2'].precision:.4f}, R={r['rouge2'].recall:.4f})")
log(f"  ROUGE-L F1  :  {r['rougeL'].fmeasure:.4f}  "
    f"(P={r['rougeL'].precision:.4f}, R={r['rougeL'].recall:.4f})")

log()
log("  Loading BERTScore model  (this may take ~30 s)…")
t0 = time.time()
from bert_score import score as bscore
P, R, F1 = bscore([pred_sum], [ref_sum], lang="en", verbose=False)
bs_p   = P.mean().item()
bs_r   = R.mean().item()
bs_f1  = F1.mean().item()
log(f"  BERTScore done in {time.time()-t0:.1f}s")
log(f"  BERTScore F1  :  {bs_f1:.4f}  (P={bs_p:.4f}, R={bs_r:.4f})")
log()

# ─────────────────────────────────────────────
# 4.  UX  —  Readability (Flesch)
# ─────────────────────────────────────────────
log("[4/4] UX READABILITY  (Flesch Metrics on Predicted Summary)")
log("-" * 40)

import textstat

fe = textstat.flesch_reading_ease(pred_sum)
fk = textstat.flesch_kincaid_grade(pred_sum)

log(f"  Flesch Reading Ease   :  {fe:.2f}")
if fe >= 80:
    label_fe = "Very Easy (5th-grade level)"
elif fe >= 60:
    label_fe = "Standard / Easy (8th-10th grade)"
elif fe >= 30:
    label_fe = "Difficult (College level)"
else:
    label_fe = "Very Difficult (Professional)"
log(f"  Interpretation        :  {label_fe}")

log(f"  Flesch-Kincaid Grade  :  {fk:.1f}  (US School Grade)")
log()

# ─────────────────────────────────────────────
# 5.  Summary Table
# ─────────────────────────────────────────────
log("=" * 60)
log("  RESULTS SUMMARY")
log("=" * 60)
log(f"  {'Metric':<32}  {'Score':>10}  Note")
log(f"  {'-'*32}  {'-'*10}  {'-'*22}")
log(f"  {'OCR — Word Error Rate':<32}  {wer:>10.4f}  lower is better")
log(f"  {'OCR — Accuracy':<32}  {accuracy:>9.2f}%  higher is better")
log(f"  {'NER — Precision':<32}  {precision:>10.4f}  higher is better")
log(f"  {'NER — Recall':<32}  {recall:>10.4f}  higher is better")
log(f"  {'NER — F1-Score':<32}  {f1_ner:>10.4f}  higher is better")
log(f"  {'ROUGE-1 F1':<32}  {r['rouge1'].fmeasure:>10.4f}  higher is better")
log(f"  {'ROUGE-2 F1':<32}  {r['rouge2'].fmeasure:>10.4f}  higher is better")
log(f"  {'ROUGE-L F1':<32}  {r['rougeL'].fmeasure:>10.4f}  higher is better")
log(f"  {'BERTScore F1':<32}  {bs_f1:>10.4f}  higher is better")
log(f"  {'Flesch Reading Ease':<32}  {fe:>10.2f}  higher = easier")
log(f"  {'Flesch-Kincaid Grade':<32}  {fk:>10.1f}  lower = simpler text")
log("=" * 60)
log()
log("  Human Evaluation Template")
log("  --------------------------")
log("  1. Medical Accuracy (1-5)  : ____")
log("     Did the AI hallucinate or drop important details?")
log("  2. Readability & Clarity (1-5) : ____")
log("     Was the explanation easy for a non-medical person to understand?")
log("  3. Overall Usefulness (1-5)   : ____")
log("     Would a patient find this summary helpful?")
log()
log("  Notes/Comments: ________________________________________")
log("=" * 60)

# ─────────────────────────────────────────────
# Write to eval_results.txt
# ─────────────────────────────────────────────
with open("eval_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\nResults saved to eval_results.txt", file=sys.__stdout__)
