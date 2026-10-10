# BanglaDiaHallu

Data and code for the article

> **BanglaDiaHallu: Hallucinations of large language models and the effect of guideline-grounded retrieval-augmented generation in Bangla diabetes question answering**
> Kamrul Hasan Nahid, Mahfuz Anam, Md. Tanvir Hossain Saon, Omma Hafsa Any, K. M. Rockybul Hassan, Md. Musfique Anwar

Forty Bangla patient questions about diabetes were each answered twice by four locally run open-weight language models (Qwen3.5-9B, TigerLLM-9B, Qwen3.6-35B-A3B, Gemma-4-26B-A4B) and by **BanglaDiaRAG**, which adds hybrid dense (BGE-M3) and sparse (BM25) retrieval from a physician-reviewed Bangla knowledge base built on Bangladesh's *National Guideline on Diabetes Mellitus* (2023) to Qwen3.5-9B. Two physicians rated all 400 responses for medical accuracy, completeness, safety, readability and reliability (0-1). A response with medical accuracy below 0.70 was classified as hallucinated.

This repository contains every question, every model response (unedited), the expert ratings, the generation code and the analysis code that reproduces all numbers, tables and result figures of the article.

## Repository structure

```
BanglaDiaHallu/
├── data/
│   ├── questions.csv                       40 questions: Bangla text, recorded variants, English translation, sub-domain
│   ├── model_responses.csv                 all 400 responses (5 configurations x 40 questions x 2 generations)
│   ├── Scoring_Table_BanglaDiaHallu.xlsx   expert ratings (consensus of two physicians)
│   ├── system_prompts/                     verbatim system prompts (baselines, BanglaDiaRAG)
│   ├── knowledge_base/                     curated Bangla knowledge base (Markdown)
│   └── original_documents/                 the original Word files (responses, prompts, questions)
├── src/
│   ├── Knowledge_Base_Construction.py      chunking, metadata, BGE-M3 + Chroma index (Fig 2A)
│   ├── BanglaDiaRAG_Pipeline.py            hybrid retrieval + generation with Qwen3.5-9B (Fig 2B)
├── analysis/
│   ├── Statistical_Analysis.py             all statistics
│   └── make_tables.py                      LaTeX rows of Tables 4-6
├── tools/
│   └── extract_responses_from_docx.py      Model Responses.docx -> data/model_responses.csv
├── results/                                analysis outputs (CSV)
├── figures/                                Figs 1-5 and S1 Fig (TIFF, 600 dpi; PDF)
├── requirements-generation.txt             environment used to build the knowledge base and run BanglaDiaRAG
└── requirements-analysis.txt               environment used for the statistical analysis
```

## Data

### `data/questions.csv`

| Column | Description |
|---|---|
| `question` | Question number (1-40) |
| `domain` | Clinical sub-domain: Dietary (1-9), Medication and insulin (10-16), Lifestyle and exercise (17-21), Complications and safety (22-26), Symptoms and early detection (27-32), Special situations (33-40) |
| `question_bn` | Question text as recorded for Qwen3.5-9B, Qwen3.6-35B-A3B, Gemma-4-26B-A4B, TigerLLM-9B and BanglaDiaRAG in the response document |
| `question_en` | English translation |

### `data/model_responses.csv`

| Column | Description |
|---|---|
| `config` | Qwen3.5-9B, TigerLLM-9B, Qwen3.6-35B-A3B, Gemma-4-26B-A4B, and BanglaDiaRAG |
| `question` | Question number (1-40) |
| `iteration` | Generation 1 or 2 ("Answer 1" / "Answer 2" in the response document) |
| `domain` | Clinical sub-domain |
| `question_text_recorded` | Question text as recorded above the response |
| `response` | Model response, unedited. Reasoning traces were removed at generation time; BanglaDiaRAG responses end with the source list (সূত্র) appended by the pipeline |

The file is produced from `data/original_documents/Model Responses.docx` by `tools/extract_responses_from_docx.py`.

### `data/Scoring_Table_BanglaDiaHallu.xlsx`

Rows 3-82 hold question 1-40 x response 1-2. Each configuration occupies seven columns: medical accuracy and completeness (two merged cells each), safety, readability and reliability. Reliability compares the two generations of a question and is entered once per question and configuration (on the response-1 row). All scores range from 0.00 to 1.00; 0.00-0.69 denotes a clinically compromised response and 0.70-1.00 an acceptable one. The ratings are the consensus of the two physicians, who rated the pooled responses without model labels, in random order.

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt
python analysis/Statistical_Analysis.py        # writes results/*.csv and figures/*
python analysis/make_tables.py > results/latex_table_rows.tex
```

The analysis uses the question as the unit of analysis (the two generations are averaged): Friedman tests with Kendall's W; Wilcoxon signed-rank tests of BanglaDiaRAG against each baseline with Holm adjustment across 20 tests, bootstrap confidence intervals of the mean paired difference (10,000 resamples of questions) and matched-pairs rank-biserial correlations; hallucination rates with Wilson and question-level cluster-bootstrap intervals and paired sign-flip permutation tests. Sensitivity analyses: linear mixed model with a random intercept for question, one-way ANOVA with Tukey's test, and thresholds from 0.50 to 0.80. The random seed is fixed (20260929), so the outputs are identical on every run.

| Output | Content |
|---|---|
| `descriptives.csv` | Mean, SD, median and quartiles per configuration and criterion (Table 4) |
| `friedman.csv` | Friedman chi-square and Kendall's W (Table 4) |
| `pairwise_rag_vs_baselines.csv` | BanglaDiaRAG vs each baseline (Table 6, Fig 5) |
| `pairwise_all_configurations.csv` | All pairwise comparisons (S2 Table) |
| `hallucination_rates.csv`, `hallucination_rag_vs_baselines.csv` | Hallucination and unsafe-response rates and comparisons (Table 5, Fig 4A) |
| `subdomain_breakdown.csv` | Hallucinations by clinical sub-domain (Fig 4B) |
| `threshold_sensitivity.csv` | Hallucination rates at thresholds 0.50-0.80 (S1 Fig) |
| `sensitivity_mixed_model.csv`, `sensitivity_oneway_anova.csv`, `sensitivity_tukey_hsd.csv` | Sensitivity analyses (S3 Table) |
| `run_to_run_agreement.csv` | Agreement in hallucination status between generations (S3 Table) |
| `script_mixing.csv`, `script_mixing_responses.csv` | Responses containing scripts other than Bangla and Latin (S3 Table) |
| `tidy_responses.csv`, `tidy_reliability.csv` | Ratings in long format (S1 Dataset) |

## Reproducing the generation

Generation needs the model weights (GGUF, Q4_K_M quantisation) and is stochastic (temperature 0.5, no fixed seed), so regenerated responses will differ in wording from the published ones. The published responses are those in `data/model_responses.csv`.


### Baselines (LM Studio)

The four baselines (Qwen3.5-9B, TigerLLM-9B, Qwen3.6-35B-A3B and Gemma-4-26B-A4B) were run interactively in the LM Studio desktop application, which uses the llama.cpp inference engine.
For each model:

1. The Q4_K_M GGUF weights were loaded in LM Studio.
2. The shared baseline system prompt was entered in the system prompt field.
3. The temperature was set to 0.5 and the context length to 8,192 tokens.
4. Response length was not limited. The context overflow policy (truncate middle) and the number of CPU threads were left at the application defaults, as were all other sampling parameters.
5. The reasoning ("thinking") mode was switched off for every model that offers it (Qwen3.5-9B, Qwen3.6-35B-A3B and Gemma-4-26B-A4B). TigerLLM-9B has no such mode.

Each question was entered as the first message of a new chat, so no conversation history was available. The response was recorded and the chat deleted, and the same question was then entered in a new chat to obtain the second, independent generation. No random seed was fixed. This was repeated for all 40 questions with each model, giving 80 responses per baseline.

Hardware: NVIDIA GeForce RTX 3070 (8 GB VRAM), 32 GB system memory.



### BanglaDiaRAG

```bash
pip install -r requirements-generation.txt
# 1. Build the knowledge base (BGE-M3 dense index in Chroma; chunk export for auditing)
python src/Knowledge_Base_Construction.py --kb-file data/knowledge_base/bangla_guidelines_new.md
# 2. Answer the 40 questions twice
python src/BanglaDiaRAG_Pipeline.py --model models/<Qwen3.5-9B Q4_K_M>.gguf --runs 2
```

Settings (identical to the study): chunks of at most 1,200 characters with 100 characters of overlap, split at blank lines, line breaks, the Bangla danda (।) and spaces, each prefixed with its section header; BGE-M3 embeddings (normalised; Chroma default L2 space, equivalent to cosine ranking) and BM25 (whitespace tokens, k1 = 1.5, b = 0.75); top 3 chunks from each retriever, merged and de-duplicated (3-6 chunks); ChatML prompt with the BanglaDiaRAG system prompt and a pre-filled closed reasoning block; Qwen3.5-9B through llama-cpp-python via LangChain with temperature 0.5, context 8,192 tokens, at most 8,192 new tokens and no fixed seed; the source list is appended from chunk metadata.


## License

Code: MIT License (see `LICENSE`).
