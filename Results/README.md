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
> **Goal:** Identify biomedical entities (`d4data/biomedical-ner-all`). Measured via strict set-based matching.

| Metric | Score | Note |
|---|---|---|
| **Precision** | **0.2500** | ↑ Higher is better |
| **Recall** | **0.3333** | ↑ Higher is better |
| **F1-Score** | **0.2857** | ↑ Higher is better |

*Note: These low scores reflect the deliberately noisy OCR inputs used in the evaluation dataset. Upstream OCR errors directly degrade downstream NER performance.*

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
