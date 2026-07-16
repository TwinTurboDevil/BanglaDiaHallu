# BanglaDiaHallu: Evaluating Hallucinations in Bangla Diabetes QA

This repository contains the official dataset, evaluation framework, and anonymized physician scoring matrices for the paper: **"BanglaDiaHallu: Evaluating Hallucinations in LLMs and the Impact of Guideline-Grounded RAG on Trustworthy Bangla Diabetes Question Answering"**.

## 📌 Project Overview
BanglaDiaHallu is a curated evaluation framework designed to assess factual consistency and clinical safety in low-resource medical NLP. It features **BanglaDiaRAG**, a hybrid dense/sparse Retrieval-Augmented Generation pipeline grounded in the official *National Guideline on Diabetes Mellitus (Bangladesh, 2023)*.

## 📂 Repository Structure
* `/dataset` : Contains the 40 validated Bangla diabetes FAQs across 6 clinical domains.
* `/evaluation_data` : The anonymized dual-physician consensus scoring matrix (Medical Accuracy, Completeness, Safety, Readability, and Reliability).
* `/src` : Core architecture components including custom Regex markdown chunking and dual-index retrieval routing (ChromaDB + BM25).

## 📊 Statistical Validation
The experimental results are mathematically validated using:
* One-Way Analysis of Variance (ANOVA) to assess cross-model performance variance.
* Tukey's Honestly Significant Difference (HSD) for post-hoc pairwise comparison.
* 95% Confidence Intervals (CI) bounding the Critical Hallucination Rate.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
