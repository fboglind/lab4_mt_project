## WorkLog

* Checked translation freqs:	
  * of: 10289 the: 8523 and: 522 patients: 408 with: 252 literature: 174 on: 172 data: 167 to: 117 in: 111 were: 102

* **Checklist:**

  - [x] Apply Tokenization Only to Russian(apply_sentencepiece.py)
  - [x] Check Language Direction Consistency (preprocess_wmt22.py) (head -2 data/parallel_corpus.tsv)
  - [x] Ensure Correct SentencePiece Model is Used at Inference 

* New run

* Train sp-model

  * update params (`sentencepiece_train.py`):

    ```
    spm.SentencePieceTrainer.train(
        input="data/only_russian_text.txt",  # Extract just Russian sentences
        model_prefix="models/spm_russian_only",
        vocab_size=24000,  # Adjust to match dataset size
        character_coverage=0.9995,  # Keep Russian coverage high
        model_type="unigram",
        input_sentence_size=0,  # Use all available data
        shuffle_input_sentence=True
    )
    ```

  - [x] sp-model trained (spm_ru_only.model)

- [x] Apply SentencePiece to Training Data to create tokenized spm-model

  * `python3 scripts/apply_sentencepiece.py --input data/russian_corpus.tsv --output data/spm_russian_corpus.tsv --model models/spm_ru_only.model`

  * the spm-model is now monolingual (ru) - saved as `models/spm_ru.model`

  * 50 sentence tokenized test set saved - `data/test_preprocessed_ru.txt`

- [ ] Train model

  * run `python3 scripts/seq2seq_train.py`
  * 



__________________

- [x] The **cleaning step** (`clean_parallel_corpus.py`) should be explicitly mentioned **before** tokenization.
- [ ] Training now **loads `sentence_aligned_corpus.tsv` instead of `parallel_corpus.tsv`**
- [ ] The `DataLoader` is correctly set up for **shorter input sequences**.
- [ ] Create reference set in notebook
- [ ] Add a **section in the notebook** that runs: python3 scripts/evaluate_translations.py --hyp data/wmt_test_translations_with_beam.txt --ref data/test_reference_en.txt
- [ ] Adjust params

Orig:

Training settings    hidden_size = 256    num_epochs = 15    learning_rate = 0.0005    batch_size = 32    teacher_forcing_ratio = 0.5    max_length = 512

Current:


\# Training settings

​    hidden_size = 256

​    num_epochs = 15

​    learning_rate = 0.0005

​    batch_size = 32

​    teacher_forcing_ratio = 0.5

​    max_length = 512

Suggested:

hidden_size = 384
num_epochs = 20  # More epochs since we process smaller units
learning_rate = 0.0005  # Keep stable learning rate
batch_size = 64  # Increase batch size for efficiency
teacher_forcing_ratio = 0.5  # Slightly reduce reliance on teacher forcing
max_length = 256  # Shorter max length for sentences

**Chosen:**

​    **\# Training settings**

​    **hidden_size = 256**

​    **num_epochs = 15**

​    **learning_rate = 0.0005**

​    **batch_size = 32**

​    **teacher_forcing_ratio = 0.5**

​    **max_length = 256**



## Paperspace

Create VM:

	* CORE Virtual Machine RTX4000
	* instance id: pskc8ozfgqys
	* ssh paperspace@64.62.255.106

* scp parallel_corpus.tsv paperspace@pskc8ozfgqys:/home/paperspace/lab4_mt_project/data

* scp ./cleaned_parallel_corpus.tsv paperspace@64.62.255.106:/home/paperspace/lab4_mt_project/data

Chats:

https://chatgpt.com/c/67d6f249-12f8-800e-bf02-088aa6fc3e84