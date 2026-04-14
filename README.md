# RAG-Powered HR Policy Q&A — DPR + FAISS + GPT-2

> **NLP · Retrieval-Augmented Generation · Dense Passage Retrieval · FAISS · GPT-2 · HuggingFace Transformers**

---

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG) question-answering system** for internal company HR policy documents — fully open-source, no external API required. Given a natural-language question such as *"What is our mobile policy?"*, the system retrieves the most relevant policy paragraphs and generates a grounded natural-language answer.

The pipeline uses **Dense Passage Retrieval (DPR)** from Facebook AI — a dual-encoder architecture trained specifically for open-domain QA — combined with a **FAISS** vector index for efficient nearest-neighbour search, and **GPT-2** as the generation model.

---

## Architecture

```
Company policy documents (.txt)
    ↓  Paragraph segmentation
    ↓  DPR context encoder (facebook/dpr-ctx_encoder-single-nq-base)
    ↓  768-dim dense embeddings → FAISS IndexFlatL2
                              ↑
User question                 │  L2 nearest-neighbour search
    ↓  DPR question encoder   │
    ↓  768-dim query embedding ┘  → Top-K relevant paragraphs
    ↓  Concatenate question + context
    ↓  GPT-2 generation → natural language answer
```

---

## Key Results

### RAG vs. No-RAG Comparison

| Approach | Answer to "What is the mobile policy?" |
|---|---|
| GPT-2 alone | Generic hallucinated text — no grounding in actual policy |
| **DPR + FAISS + GPT-2 (RAG)** | Accurate, policy-faithful answer drawn from retrieved paragraphs |

### Embedding Validation

t-SNE visualisation of DPR embeddings confirms the context encoder learns meaningful semantic structure: related policy topics cluster together (health & safety, anti-discrimination), while distinct sections occupy separate embedding regions.

---

## Technical Highlights

- **Dual-encoder DPR** — Question and context encoders trained jointly on QA pairs, producing directly comparable embeddings without fine-tuning
- **FAISS flat L2 index** — Exact nearest-neighbour search; O(1) updates when adding new policy documents; scalable to millions of vectors with approximate index variants
- **With vs. without context experiment** — Explicitly demonstrates how RAG grounds generation and eliminates hallucination compared to pure language model inference
- **Generation parameter study** — Systematic comparison of `num_beams`, `max_new_tokens`, `min_length`, and `length_penalty` shows their impact on answer quality
- **Fully local, no API keys** — All models (`dpr-ctx_encoder`, `dpr-question_encoder`, `gpt2`) are downloaded from HuggingFace Hub and run on CPU

---

## Dataset

- **Source:** Synthetic company HR policy document (20 policies including Code of Conduct, Leave Policy, Mobile Phone Policy, Drug & Alcohol Policy, Anti-Discrimination Policy, etc.)
- **Format:** Plain text, paragraph-segmented
- **Corpus size:** ~70 paragraphs
- **Embedding dimension:** 768 (DPR pooler output)

---

## Project Structure

```
.
├── RAG_DPR_FAISS_GPT2_HRPolicy_Portfolio.ipynb   # Main notebook
├── README.md
└── companyPolicies.txt                            # Policy corpus (auto-downloaded)
```

---

## How to Run

```bash
pip install wget transformers faiss-cpu matplotlib scikit-learn torch

jupyter notebook RAG_DPR_FAISS_GPT2_HRPolicy_Portfolio.ipynb
```

All model weights are downloaded automatically from HuggingFace Hub on first run (~900MB total). No API keys required.

---

## Skills Demonstrated

`HuggingFace Transformers` · `DPR` · `FAISS` · `GPT-2` · Dual-Encoder Retrieval · Dense Passage Retrieval · Vector Indexing · RAG · `t-SNE` · NLP · Generation Parameter Tuning · Open-Domain QA

