# Log/Readme

### Local

**Directories:**

* lab4b/ (copied from server and uploaded to GitHub repo)
  * bpe_model.codes
  * bpe_parallel_corpus.tsv
  * bpe_train_apply.py
  * bpe_training_data.txt
  * model_checkpoint.pt
  * README.md
  * seq2seq_attentionOLD.py
  * seq2seq_model.py
  * seq2seq_train.py
  * test_preprocessed_en.txt
  * train_mt_OLD.py
  * translate_wmt_testOLD.py
  * translate_wmt_test.py
  * wmt_test_translations.txt



* **lab4/**
  * abstracts/
  * pycache/
  * test/
  * trainWmt22/
    * **en_ru_pmids.txt**
    * train22_eng_chi.txt
    * train22_eng_fre.txt
    * train22_eng_ger.txt
    * train22_eng_ita.txt
    * train22_eng_por.txt
    * train22_eng_spa.txt
  * bpe_model.codes
  * bpe_parallel_corpus.tsv
  * bpe_train_apply.py
  * bpe_training_data.txt
  * clean_and_tokenize.py
  * cleaned_parallel_corpus.tsv
  * en2ru_en.txt
  * parallel_corpus_sample.tsv
  * parallel_corpus.tsv
  * preprocess_test_set.py
  * preprocess_wmt22.py
  * requirements.txt
  * ru2en_ru.txt
  * seq2seq_attention.py
  * seq2seq_model.py
  * seq2seq_train.py
  * test_preprocessed_en.txt
  * test_preprocessed_ru.txt
  * train_mt_OLD.py
  * translate_wmt_test.py
  * upload_to_server
  * wmt24_biomedical_task.md
  * wmtbio22_train_data.py



**Log:**

*  `wmtbio22_train_data.py` - downloads data (abstracts  from PubMed (Medline))

  * `python wmtbio22_train_data.py pmid_list.txt output_dir/`

* installed some dependencies: biopython (not needed?)

* Extract the dataset (`trainWmt22.zip`).  file contains PubMed IDs (PMIDs) rather than the actual training data

* 

* *Decided to use pubmed data even though this breaks WMT-rules*

* ran `python wmtbio22_train_data.py en_ru_pmids.txt abstracts/`

* changed wmtbio22_train: 

  ```
  # Replaced with custom function
  def get_lang1_lang2(filename):
      return "en", "ru"  # Manually set language pair
  
  ```

  

### Preprocessing the Data

* `preprocess_wmt22.py` pairs the abstracts and save them in a single file:
  * `parallel_corpus.tsv` a parallel corpus with English-Russian sentence pairs in tab-separated format:
* *Tried NLTK but switched to Moses, the switched to SentencePiece*
* ~~Cleaning & Tokenization~~
  * ~~using `clean_and_tokenize.py`~~
    * ~~*decided on BPE (subword-nmt)* SentencePiece was an option~~
  * ~~ran `bpe_train_apply.py`~~
    * ~~output saved to`bpe_parallel_corpus.tsv`.~~



*Compared test data (Gold set from WMT) with training data and ascertained that the abstracts differed in length* - It was decided not to alter any of the sets

* prepared (ran) `preprocess_test_set.py`

  got:

  * test_preprocessed_en.txt (Moses-tokenized + BPE-encoded English abstracts)
  * test_preprocessed_ru.txt

### Preparing Inference script

* Created `translate_wmt_test.py`
  * output: `wmt_test_translations.txt`



### Preparing training script

* preprared (ran) ``train_mt.py`
  * this script was NOT used instead

*The old attention_seq file was refactored*, these files were the ones eventually used:

* `seq2seq_train.py` (and `seq2seq_model.py`)

*This project structure was decided on*:

```
/project_directory/
│── seq2seq_model.py         # Model architecture (Encoder, Decoder)
│── seq2seq_train.py         # Training script
│── seq2seq_inference.py     # Inference script
│── data/                    # Data folder (optional)
│── checkpoints/             # Model checkpoints
│── utils.py                 # Helper functions (optional)
```





*The parameters were reviewed*

*Padding was added to fix mismatched shapes*:

* Finds the longest sequence in the batch.

* Pads all other sequences with `pad_idx=0`** to match that length.

* Uses `torch.stack()` instead of `torch.cat()`** to correctly handle padding.

*Experienced more problems relating to shape, but these were fixed*

*Training and translation was ran on the server*

The output had some problem relating to the encoding/detokenization







