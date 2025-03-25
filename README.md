# 🧬 Russian-to-English Biomedical Machine Translation with PyTorch

This project implements a **Neural Machine Translation (NMT)** system for the **Russian → English** biomedical domain, developed using **PyTorch**. It is designed for the [WMT24 Biomedical Shared Task](https://www2.statmt.org/wmt24/biomedical-translation-task.html), translating scientific abstracts from Medline.

The system uses a **sequence-to-sequence (seq2seq) architecture with attention**, supports both **GRU and LSTM**, and applies **SentencePiece** subword tokenization. Evaluation is done with **BLEU**, **chrF**.

---

## Pipeline Overview

### Data
- WMT22 Biomedical Russian-English abstracts (parallel data)

### Preprocessing
- Text cleaning
- Sentence splitting
- SentencePiece tokenization

### Model
- Attention-based encoder-decoder (GRU or LSTM)
- Trained with teacher forcing and gradient accumulation
- Built from scratch in PyTorch

### Inference
- Translates tokenized Russian input back to natural English

### Evaluation
- BLEU and chrF scores via sacreBLEU
- COMET planned

---

## 📂 Directory Structure (approx.)

```
.
├── abstracts     [EMPTY]                 # Folder with medical abstracts                    
├── data          [EMPTY]                 # Folder for dataset & processed files                 
├── lab4.ipynb                            # Jupyter notebook. Use this to run everything
├── models        [EMPTY]                 # Trained models are saved here                          
├── README.md                             # This file
├── requirements.txt                      # Use this to create environment for project
├── run_training.sh                       # To train on server
├── scripts/                               # Folder containing all scripts
│    ├── abstract_analysis.py              # Basic analysis of files in /abstracts directory
│    ├── advanced_evaluation.py
│    ├── apply_sentencepiece.py            # Tokenize dataset using SentencePiece
│    ├── clean_parallel_corpus.py          # Cleaning by removing misaligned abstracts
│    ├── clean_russian_openers.py          # Removes openers and headers in ru text
│    ├── evaluate_translations.py          # An evaluation script
│    ├── extract_frequent_numericals.py    # Extracts numericals
│    ├── extract_numbers_and_latin.py      # Extract numericals and latin chars
│    ├── extract_references.py             # Extracts reference translations for eval
│    ├── extract_russian.py                # Extracts ru-only corpus
│    ├── preprocess_wmt22.py               # Data extraction script
│    ├── sentencepiece_train.py            # Train SentencePiece tokenizer
│    ├── seq2seq_model.py                  # PyTorch seq2seq model
│    ├── seq2seq_train.py                  # Training script
│    ├── split_abstracts.py                # Splits abstract into sentences
│    ├── split_single_column.py            # Split a single-language file into sentences
│    ├── translate_wmt_test.py             # Inference script (translation)
│    └── word_counter.py                   # Counts words

```


## 1. Preprocessing Pipeline

### 1.1 Extract Training & Test Data

```bash
python3 scripts/preprocess_wmt22.py
```
---
### 1.2 Sentence Splitting

python3 scripts/split_abstracts.py \
  --input data/test_parallel.tsv \
  --output data/test_parallel_sentences.tsv

cut -f1 data/test_parallel_sentences.tsv > data/test_raw_ru_sentences.txt
---

### 1.3 Train SentencePiece Tokenizer
```bash
python3 scripts/sentencepiece_train.py
```
This generates:
- **`spm_ru_en.model`** (SentencePiece model).
- **`spm_ru_en.vocab`** (Vocabulary file).
---
### 1.4 Apply SentencePiece
```bash
python3 scripts/apply_sentencepiece.py \
  --input data/parallel_corpus.tsv \
  --output data/spm_parallel_corpus.tsv

python3 scripts/apply_sentencepiece.py \
  --input data/test_raw_ru_sentences.txt \
  --output data/test_preprocessed_ru_sentences.txt
```
## 2. Train the Model
```
python3 scripts/seq2seq_train.py \
  --train-file data/spm_parallel_corpus.tsv \
  --batch-size 8 \
  --accum-steps 4 \
  --model-type lstm \
  --checkpoint models/model_checkpoint.pt
```
Model checkpoint is saved to models/model_checkpoint.pt

## 3. Translate the Test Set
```bash
python3 scripts/translate_wmt_test.py
```
This uses:

    data/test_preprocessed_ru_sentences.txt

    Outputs: data/wmt_test_translations.txt

## 4. Extract Reference Translations
```bash
python3 scripts/extract_references.py \
  --parallel data/test_parallel_sentences.tsv \
  --test data/test_raw_ru_sentences.txt \
  --output data/test_reference_en.txt \
  --src-col Russian \
  --tgt-col English
```

## 5. Evaluate Translations
```bash
python3 scripts/evaluate_translations.py \
  --hyp data/wmt_test_translations.txt \
  --ref data/test_reference_en.txt \
  --output data/eval_results.txt
```
Outputs:

    BLEU score

    chrF score

    eval_results.txt with detailed breakdown


