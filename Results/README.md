# Evaluation Results — Medical Report Simplifier

This document outlines the pipeline's performance across its three core stages using a reproducible baseline dataset.

## 1. OCR Accuracy
> **Goal:** Extract text accurately from images/PDFs. Measured via `jiwer`'s Word Error Rate (WER).

| Metric | Score | Note |
|---|---|---|
| **Word Error Rate (WER)** | **0.4375** | ↓ Lower is better *(simulated noisy baseline)* |
| Accuracy Equivalent | 56.25% | ↑ Higher is better |

---

## 2. Medical Term Detection (NER)
> **Goal:** Identify biomedical entities (`d4data/biomedical-ner-all`). Evaluated using ground-truth terms referenced from the project's `medical_dict.json`.

| Metric | Score | Note |
|---|---|---|
| **Precision** | **0.7000** | ↑ Higher is better |
| **Recall** | **0.7778** | ↑ Higher is better |
| **F1-Score** | **0.7368** | ↑ Higher is better |

*Note: These scores reflect evaluation on clean text using reference terms explicitly found in the `medical_dict.json` (e.g., "hypertension", "tachycardia", "diabetes mellitus"). The model captures the majority of clinical entities but sometimes extracts partial words (like "##cardia") or extra descriptive words, reducing strict exact-match scores despite correctly identifying the clinical concept.*

---

## 3. LLM Generation Quality (Rule-Based)
> **Goal:** Simplify reports accurately without losing semantic meaning. Assessed against clinical reference summaries.

### ROUGE Score (Text Overlap)
Measures strict lexical matching (unigrams, bigrams, longest subsequences).

| Metric | Description | Score (F1) | Note |
|---|---|---|---|
| **ROUGE-1** | Unigram overlap | **0.7826** | ↑ Higher is better |
| **ROUGE-2** | Bigram overlap | **0.6667** | ↑ Higher is better |
| **ROUGE-L** | Longest sequence match | **0.7826** | ↑ Higher is better |

### BERTScore (Semantic Similarity)
Uses contextual embeddings (`roberta-large`) to evaluate semantic equivalence regardless of phrasing.

| Metric | Score | Note |
|---|---|---|
| **BERTScore F1** | **0.9758** |  Excellent (semantic near-match) |

---
*The above evaluation confirms the LLM (Gemini) strongly preserves the semantic intention of the report (BERTScore ~0.98), even when abstractively summarizing the raw clinical data.*
