# Russian-to-English Machine Translation with PyTorch

This project implements a **Neural Machine Translation (NMT)** system for the Russian → English biomedical domain, developed using **PyTorch**. It uses data from the WMT24 Biomedical Translation Shared Task and translates scientific abstracts from Medline.
The system is based on a **sequence-to-sequence (seq2seq)** GRU or LSTM architecture with attention, and uses **SentencePiece** for subword tokenization. It supports training on sentence-aligned biomedical data, inference on raw abstracts, and evaluation with BLEU and chrF,
---

## Project Overview [To be updated] - lab4.ipynb is the most up-to-date version.

### 1️⃣ Data Collection
- We use **WMT22 Biomedical Training Data** for **Russian-English** translation.
- The dataset consists of **parallel medical abstracts**.

### 2️⃣ Preprocessing
- **Text cleaning & normalization**.
- **Tokenization** using **SentencePiece**
- **Conversion to tensors** for PyTorch training.

### 3️⃣ Model Training
- **Encoder-Decoder GRU model with attention**.
- **Mini-batch training** with **gradient clipping**.
- **Teacher forcing to stabilize training**.

### 4️⃣ Inference (Translation)
- Translate **Russian test abstracts** into **English**.
- Use **SentencePiece for decoding**.

### 5️⃣ Evaluation (To Be Added)
- Compute **BLEU, chrF** on test translations.

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
│    ├── split_single_column.py            # Split a single-language file into sentences 
│    ├── split_abstracts.py                # Splits abstract into sentences
│    ├── translate_wmt_test.py             # Inference script (translation)
│    └── word_counter.py                   # Counts words

```

---

## 1. Data Collection

1. Download the WMT22 dataset and extract parallel Russian-English abstracts.
2. Run `preprocess_wmt22.py` to create the file `parallel_corpus.tsv` (tab-separated, Russian ↔ English).
3. (Optional) Run analysis: `python3 scripts/abstract_analysis.py --input data/parallel_corpus.tsv`
4. Clean the dataset: `python3 scripts/clean_parallel_corpus.py --input data/parallel_corpus.tsv --output data/cleaned_parallel_corpus.tsv`
4. Split the paralell corpus into sentences: 
`python3 scripts/split_abstracts.py --input data/cleaned_parallel_corpus.tsv --output data/sentence_aligned_corpus.tsv`
5. For uni-language spm training, extract Russian sentences only: `cut -f 1 data/sentence_aligned_corpus.tsv > data/russian_sentences_corpus.tsv`

---

## 2. Preprocessing

### 1️⃣  Train SentencePiece Tokenizer

Run the following script to **train SentencePiece on the dataset**:

```bash
python3 scripts/sentencepiece_train.py
```

This generates:
- **`spm_ru_en.model`** (SentencePiece model).
- **`spm_ru_en.vocab`** (Vocabulary file).

### 2️⃣ Apply SentencePiece to Training Data

Tokenize the dataset for model training:

```bash
~~python3 scripts/apply_sentencepiece.py --input data/parallel_corpus.tsv --output data/spm_parallel_corpus.tsv --is_parallel~~
`python3 scripts/apply_sentencepiece.py --input data/parallel_corpus.tsv --output data/spm_parallel_corpus.tsv`
```

This creates:
~~- **`spm_parallel_corpus.tsv`** (Tokenized dataset).~~
**`spm_russian_corpus.tsv`** (Tokenized dataset)

### 3️⃣ Prepare the Test Data

Run the same script again:

```bash
python3 scripts/apply_sentencepiece.py --input data/test_raw_ru.txt --output data/test_preprocessed_ru.txt
python3 scripts/apply_sentencepiece.py --input data/test_raw_ru.txt --output data/test_preprocessed_ru.txt

```

---

## 3. Train the Model

Run the following command to train the model:

```bash
python3 scripts/seq2seq_train.py
```

### Training Details
- Model is trained for **15 epochs**.
- Uses **mini-batches (batch size = 32)**.
- Learning rate: **0.0005** (with Adam optimizer).
- **Gradient clipping** prevents exploding gradients.

After training, the model is saved as:

```
models/model_checkpoint.pt
```

---

## 4. Run Inference (Translate Russian to English)

To translate Russian medical abstracts:

```bash
python3 scripts/translate_wmt_test.py
```

This generates:

```
wmt_test_translations.txt  # Translated English abstracts
```



##  5. Evaluation (To Be Added)

Coming soon: **BLEU, chrF, COMET** evaluation.

---

## Notes

- The system **only translates Russian → English**.
- Uses **SentencePiece for tokenization**, avoiding Moses/BPE issues.
- Implements **Attention-based seq2seq model** in PyTorch.

---

## To-Do
..almost done

---

### Credits

- WMT Biomedical Translation Task  
- PyTorch seq2seq model adapted for **Russian-English MT**  

---

### 🔹 Final Notes

This README provides step-by-step guidance from **data collection** → **preprocessing** → **training** → **translation**. 
