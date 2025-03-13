# Russian-to-English Machine Translation with PyTorch

This repository implements a **Neural Machine Translation (NMT) model** using **Sequence-to-Sequence with Attention** in **PyTorch**. The project follows the **WMT Biomedical Translation Task**, training a **Russian-to-English (ru→en) translation model**.

We use **SentencePiece for tokenization**, a **GRU-based seq2seq model with attention**, and train the model on **WMT22 biomedical data**.

---

## Project Overview

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
- Compute **BLEU, chrF, COMET** on test translations.

---

## 📂 Directory Structure (intended)

```
├── data/                          # Folder for dataset & processed files
│   ├── parallel_corpus.tsv        # Original parallel dataset
│   ├── spm_parallel_corpus.tsv    # Tokenized dataset using SentencePiece
│   ├── test_preprocessed_ru.txt   # Tokenized test data (Russian)
├── models/
│   ├── spm_ru_en.model            # SentencePiece model
│   ├── model_checkpoint.pt        # Trained model checkpoint
├── old_files/
|   ├── [...]                      # Old files to be deleted   
├── scripts/
│   ├── apply_sentencepiece.py     # Tokenize dataset using SentencePiece
│   ├── preprocess_wmt22.py        # Data extraction script
│   ├── sentencepiece_train.py     # Train SentencePiece tokenizer
│   ├── seq2seq_model.py           # PyTorch seq2seq model
│   ├── seq2seq_train.py           # Training script
│   ├── translate_wmt_test.py      # Inference script (translation)
├── README.md                      # This file
├── requirements.txt               # Lists dependencies 
```

---

## 1. Data Collection

1. Download the WMT22 dataset and extract parallel Russian-English abstracts.
2. Run `preprocess_wmt22.py` to create the file `parallel_corpus.tsv` (tab-separated, Russian ↔ English).

---

## 2. Preprocessing

### 1️⃣ Train SentencePiece Tokenizer

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
python3 scripts/apply_sentencepiece.py --input data/parallel_corpus.tsv --output data/spm_parallel_corpus.tsv --is_parallel
```

This creates:
- **`spm_parallel_corpus.tsv`** (Tokenized dataset).

### 3️⃣ Prepare the Test Data

Run the same script again:

```bash
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

✅ **Complete training pipeline**  
✅ **Switch to SentencePiece**  
✅ **Fix detokenization issues**  
      **Add evaluation metrics**  

---

### Credits

- WMT Biomedical Translation Task  
- PyTorch seq2seq model adapted for **Russian-English MT**  

---

### 🔹 Final Notes

This README provides step-by-step guidance from **data collection** → **preprocessing** → **training** → **translation**. 
