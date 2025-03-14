"""translate_wmt_test.py"""
import torch
import sentencepiece as spm
import sys
import os
from tqdm import tqdm
from seq2seq_model import EncoderRNN, AttnDecoderRNN, tensor_from_sentence

# Paths
sp_model = "models/spm_ru_en.model"  # SentencePiece model path
test_input_file = "data/test_preprocessed_ru.txt"  # Input: Russian test abstracts
output_translation_file = "data/wmt_test_translations.txt"  # Output: Translated English abstracts
checkpoint_file = "models/model_checkpoint.pt"  # Model checkpoint

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Maximum length for translation
max_length = 512

# Load SentencePiece model
print("Loading SentencePiece model...")
sp = spm.SentencePieceProcessor(model_file=sp_model)
print("SentencePiece model loaded.")

# Load model checkpoint
print(f"Loading model checkpoint from: {checkpoint_file}")
checkpoint = torch.load(checkpoint_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load model parameters
hidden_size = checkpoint["hidden_size"]
src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]
tgt_index2word = checkpoint.get("tgt_index2word", {idx: word for word, idx in tgt_vocab.items()})

# Initialize encoder and decoder models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Translating on: {device}")

encoder = EncoderRNN(len(src_vocab), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(tgt_vocab), max_length=max_length).to(device)

encoder.load_state_dict(checkpoint["enc_state"])
decoder.load_state_dict(checkpoint["dec_state"])

encoder.eval()
decoder.eval()
print("Model loaded successfully.")

def translate_sentence(sentence, encoder, decoder, src_vocab, tgt_index2word, max_length=512):
    """Translate a single sentence"""
    with torch.no_grad():
        # Preprocess input sentence
        input_tensor = tensor_from_sentence(
            src_vocab, sentence, 
            pad_idx=PAD_token, eos_idx=EOS_token, unk_idx=UNK_token
        ).unsqueeze(1).to(device)
        
        # Initialize encoder hidden state
        encoder_hidden = encoder.get_initial_hidden_state(batch_size=1)
        
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        
        # Prepare decoder input (start with SOS token)
        decoder_input = torch.tensor([SOS_token], device=device)
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        
        # Forward pass through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, encoder_outputs, decoder_hidden)
            
            # Get the top predicted token
            topv, topi = decoder_output.topk(1)
            token_idx = topi.item()
            
            # If EOS token, stop decoding
            if token_idx == EOS_token:
                break
                
            # Add token to output list
            if token_idx in tgt_index2word:
                decoded_words.append(tgt_index2word[token_idx])
            else:
                decoded_words.append("<UNK>")
            
            # Update decoder input for next step
            decoder_input = torch.tensor([token_idx], device=device)
        
        # Join words to form sentence
        return " ".join(decoded_words)

# Start translating the test set
print(f"Translating test set from {test_input_file}...")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_translation_file), exist_ok=True)

# Read test data
try:
    with open(test_input_file, "r", encoding="utf-8") as f:
        test_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(test_sentences)} test sentences.")
    
    # Translate each sentence
    with open(output_translation_file, "w", encoding="utf-8") as f_out:
        for idx, sentence in enumerate(tqdm(test_sentences)):
            translation = translate_sentence(sentence, encoder, decoder, src_vocab, tgt_index2word)
            
            # Write translation to file
            f_out.write(translation + "\n")
            
            # Print progress
            if (idx + 1) % 10 == 0:
                print(f"Translated {idx + 1}/{len(test_sentences)} sentences.")
    
    print(f"Translation complete! Output saved to {output_translation_file}")