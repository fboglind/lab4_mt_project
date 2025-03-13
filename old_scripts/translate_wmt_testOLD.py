import torch
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE
from seq2seq_model import EncoderRNN, AttnDecoderRNN, tensor_from_sentence
import os

# Paths
test_input_file = "test_preprocessed_en.txt"  # Input: English test abstracts (after Moses + BPE)
output_translation_file = "wmt_test_translations.txt"  # Output: Translated Russian abstracts
bpe_model_file = "bpe_model.codes"
checkpoint_file = "model_checkpoint.pt"  # Modify if your model is saved under a different name

# Load tokenizers
mt_en = MosesTokenizer(lang='en')
md_ru = MosesDetokenizer(lang='ru')

# Load BPE model
print("Loading BPE model...")
with open(bpe_model_file, "r", encoding="utf-8") as bpe_file:
    bpe = BPE(bpe_file)
print("BPE model loaded.")

# Load model checkpoint
print("Loading model checkpoint from:", checkpoint_file)
checkpoint = torch.load(checkpoint_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load model parameters
hidden_size = checkpoint["hidden_size"]
src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]

# Initialize encoder and decoder models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderRNN(len(src_vocab), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(tgt_vocab)).to(device)

encoder.load_state_dict(checkpoint["enc_state"])
decoder.load_state_dict(checkpoint["dec_state"])
encoder.eval()
decoder.eval()

print("Model loaded successfully.")

# Function to preprocess input text
def preprocess_text(text):
    tokenized = " ".join(mt_en.tokenize(text))  # Moses Tokenization
    bpe_encoded = " ".join(bpe.process_line(tokenized))  # Apply BPE
    return bpe_encoded

# Function to translate a single abstract
def translate_sentence(sentence, encoder, decoder, src_vocab, tgt_vocab, max_length=512):
    input_tensor = tensor_from_sentence(src_vocab, preprocess_text(sentence)).unsqueeze(1).to(device)
    input_length = input_tensor.size(0)
    
    # Initialize hidden state for batch size 1
    encoder_hidden = encoder.get_initial_hidden_state(batch_size=1)

    # Pass input through encoder
    encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=device)
    embedded = encoder.embedding(input_tensor)  # Ensure correct shape for embedding
    encoder_output, encoder_hidden = encoder.gru(embedded, encoder_hidden)

    encoder_outputs[:encoder_output.size(0)] = encoder_output

    # Decoder initialization
    decoder_input = torch.tensor([[0]], device=device)  # SOS token
    decoder_hidden = encoder_hidden
    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input.squeeze(1), encoder_outputs, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        
        if topi.item() == 1:  # EOS token
            break
        else:
            decoded_words.append(list(tgt_vocab.keys())[list(tgt_vocab.values()).index(topi.item())])

        decoder_input = topi.detach()

    return " ".join(decoded_words)

# Start translating the test set
print("Translating test set...")
with open(test_input_file, "r", encoding="utf-8") as f_in, open(output_translation_file, "w", encoding="utf-8") as f_out:
    for idx, line in enumerate(f_in):
        line = line.strip()
        if line:
            translation = translate_sentence(line, encoder, decoder, src_vocab, tgt_vocab)
            detokenized = md_ru.detokenize(translation.split())  # Detokenize to match natural output
            f_out.write(detokenized + "\n")
        
        if idx % 5 == 0:
            print(f"Translated {idx + 1} abstracts...")

print(f"Translation complete! Output saved to {output_translation_file}")
