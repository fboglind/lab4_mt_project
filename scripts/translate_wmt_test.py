import torch
import sentencepiece as spm
from subword_nmt.apply_bpe import BPE
from seq2seq_model import EncoderRNN, AttnDecoderRNN, tensor_from_sentence


# Paths
sp_model = "spm_ru_en.model"  # SentencePiece model path
test_input_file = "test_preprocessed_ru.txt"  # Input: Russian test abstracts (after SentencePiece)
output_translation_file = "wmt_test_translations.txt"  # Output: Translated English abstracts
checkpoint_file = "model_checkpoint.pt"  # Modify if your model is saved under a different name


# Load SentencePiece model
print("Loading SentencePiece model...")
sp = spm.SentencePieceProcessor(model_file=sp_model)
print("SentencePiece model loaded.")

# Load model checkpoint
print("Loading model checkpoint from:", checkpoint_file)
checkpoint = torch.load(checkpoint_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load model parameters
hidden_size = checkpoint["hidden_size"]
src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]
tgt_index2word = {index: word for word, index in tgt_vocab.items()}  # Reverse mapping

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
    return " ".join(sp.encode(text, out_type=str))

# Function to translate a single abstract
def translate_sentence(sentence, encoder, decoder, src_vocab, tgt_vocab, max_length=512):
    input_tensor = tensor_from_sentence(src_vocab, preprocess_text(sentence)).unsqueeze(1).to(device)
    
    encoder_hidden = encoder.get_initial_hidden_state(batch_size=1)
    encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=device)
    
    embedded = encoder.embedding(input_tensor)
    encoder_output, encoder_hidden = encoder.gru(embedded, encoder_hidden)
    encoder_outputs[:encoder_output.size(0)] = encoder_output

    decoder_input = torch.tensor([[0]], device=device)  # SOS token
    decoder_hidden = encoder_hidden
    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input.squeeze(1), encoder_outputs, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        
        if topi.item() == 1:  # EOS token
            break
        else:
            word = tgt_index2word.get(topi.item(), "<UNK>")  
            decoded_words.append(word)

        decoder_input = topi.detach()

    # **Fix: Properly detokenize**
    translation = sp.decode(decoded_words)  # Ensure correct SentencePiece decoding
    return translation


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
