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











## Paperspace

Create VM:

	* CORE Virtual Machine RTX4000
	* 