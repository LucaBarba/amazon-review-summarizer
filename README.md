# Amazon Review Summarizer — Multi-Model Benchmark

NLP class project benchmarking five abstractive Seq2Seq summarization models against an extractive TF-IDF baseline across 400 Amazon products from the *Toys & Games* category.

---

## Models

| Model | Type | Training Domain |
|---|---|---|
| `philschmid/bart-large-cnn-samsum` | Abstractive (Seq2Seq) | Dialogue / SAMSum |
| `facebook/bart-large-cnn` | Abstractive (Seq2Seq) | News / CNN-DailyMail |
| `sshleifer/distilbart-cnn-12-6` | Abstractive distilled | News / CNN-DailyMail |
| `google/pegasus-cnn_dailymail` | Abstractive (Seq2Seq) | News / CNN-DailyMail |
| `Falconsai/text_summarization` | Abstractive compact | General |
| TF-IDF Medoid | Extractive | — |
| `Qwen/Qwen2.5-1.5B-Instruct` | LLM (silver reference) | General instruction-tuned |

---

## Results (Overall group, 400 products)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Sentiment Align. |
|---|---|---|---|---|---|
| Qwen2.5-1.5B *(ref)* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9425 |
| **BART-SamSum** | **0.3139** | **0.0615** | **0.1924** | **0.8725** | 0.8900 |
| BART-CNN | 0.2938 | 0.0479 | 0.1743 | 0.8643 | 0.9225 |
| Falconsai | 0.3036 | 0.0500 | 0.1687 | 0.8593 | **0.9575** |
| DistilBART | 0.3006 | 0.0503 | 0.1777 | 0.8592 | 0.8700 |
| TF-IDF Extractive | 0.2554 | 0.0428 | 0.1616 | 0.8608 | 0.9550 |
| PEGASUS-CNN | 0.2368 | 0.0397 | 0.1553 | 0.8518 | 0.8200 |

---

## Key Findings

1. **Training domain matters more than model size** — BART-SamSum leads across all reference-based metrics. Dialogue fine-tuning transfers better to informal review text than news fine-tuning, despite identical architecture to BART-CNN.

2. **Extractive models preserve sentiment better** — TF-IDF (0.955) and Falconsai (0.958) outperform larger abstractive models on sentiment alignment. Abstractive models smooth over strong opinions (*abstractive smoothing*).

3. **BERTScore gaps are narrow; ROUGE gaps are wide** — all models score within 0.022 BERTScore F1 of each other (they understand content similarly), but ROUGE-2 varies 2.5x between best and worst, reflecting stylistic differences.

4. **PEGASUS is the clear outlier** — trained for extreme compression on news, it scores lowest across all metrics on informal multi-document review data.

---

## Pipeline

```
Dataset streaming (HuggingFace)
  -> 400 products selected (fixed seed=123)
    -> 8 review groups per product (overall, positive, negative, 1-5 star)
      -> TF-IDF medoid selection (top 15 reviews per group)
        -> Silver reference generation (Qwen2.5-1.5B FP16, local GPU)
        -> 5 abstractive models + extractive baseline
          -> ROUGE + BERTScore + Sentiment Alignment evaluation
```

All outputs are saved incrementally per product per model — the pipeline resumes automatically from any interruption.

---

## Setup

**Requirements:** Python 3.11, CUDA-capable GPU (6 GB+ VRAM), HuggingFace account.

```bash
# Clone
git clone https://github.com/BarbaDLuca/amazon-review-summarizer
cd amazon-review-summarizer

# Environment
python -m venv amazon_env
.\amazon_env\Scripts\activate          # Windows PowerShell
# source amazon_env/bin/activate       # macOS / Linux

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install transformers==4.52.4 datasets==2.20.0 accelerate sentencepiece tiktoken
pip install scikit-learn==1.3.2 nltk pandas==2.1.4 numpy==1.26.4 tqdm
pip install rouge-score bert-score
pip install ipykernel notebook

# Register Jupyter kernel
python -m ipykernel install --user --name=amazon_env --display-name "Python (amazon_env)"

# Launch
python -m jupyter notebook
```

Open `amazon_review_summarizer_local.ipynb`, select the **Python (amazon_env)** kernel, paste your HuggingFace token into the config cell, and run all cells.

**Expected runtime:** 4–6 hours for all 400 products × 5 models on a GTX 1660 Ti.

---

## Repository Structure

```
amazon-review-summarizer/
├── amazon_review_summarizer_local.ipynb   # Main pipeline notebook
├── interface.html                          # Model comparison interface
├── index.html                              # GitHub Pages project page
├── outputs/
│   ├── products.json                       # 400 selected ASINs (fixed)
│   ├── product_reviews.json               # Collected reviews cache
│   ├── silver_references.json             # Qwen-generated references
│   ├── summaries.csv                      # All model outputs (flat)
│   ├── evaluation_results.csv             # All metric scores
│   └── <model_name>/                      # Per-product JSON files
│       └── <asin>.json
└── README.md
```

---

## Dataset

`BarbaDLuca/amazon-reviews-2023-with-asin` on HuggingFace — a processed subset of the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset with `parent_asin` as a product key, enabling multi-document grouping by product.

---

*Luca Delpino Barbabella — NLP Class Project, PPGI/UnB 2026*
