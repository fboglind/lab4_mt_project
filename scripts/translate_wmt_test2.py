"""translate_wmt_test2.py - Script for translating test data using trained model
"""
import torch
import sentencepiece as spm
from tqdm import tqdm
import os
from seq2seq_model import GRUEncoder, GRUAttnDecoder, tensor_from_sentence


class Translator:
    """Class for translating text using a trained sequence-to-sequence model"""
    
    def __init__(self, model_path, sp_model_path, device=None, max_length=512):
        """
        Initialize translator
        
        Args:
            model_path: Path to the trained model checkpoint
            sp_model_path: Path to the SentencePiece model
            device: Device to run translation on (cpu/cuda)
            max_length: Maximum sequence length for translation
        """
        self.model_path = model_path
        self.sp_model_path = sp_model_path
        self.max_length = max_length
        
        # Determine device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Translating on: {self.device}")
        
        # Load SentencePiece model
        self.load_tokenizer()
        
        # Load translation model
        self.load_model()
    
    def load_tokenizer(self):
        """Load SentencePiece tokenizer"""
        try:
            print(f"Loading SentencePiece model from {self.sp_model_path}...")
            self.sp = spm.SentencePieceProcessor(model_file=self.sp_model_path)
            print("SentencePiece model loaded successfully")
        except Exception as e:
            print(f"Error loading SentencePiece model: {str(e)}")
            raise
    
    def load_model(self):
        """Load trained translation model from checkpoint"""
        try:
            print(f"Loading model checkpoint from {self.model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device
            )
            
            # Extract model parameters and vocabularies
            self.src_vocab = checkpoint.get("src_vocab") or checkpoint.get("src_vocab")
            self.tgt_vocab = checkpoint.get("tgt_vocab") or checkpoint.get("tgt_vocab")
            self.tgt_index2word = checkpoint.get("tgt_index2word") or {idx: word for word, idx in self.tgt_vocab.items()}
            
            # Special tokens
            self.special_tokens = checkpoint.get("special_tokens") or {
                "PAD": 0,
                "SOS": 1,
                "EOS": 2,
                "UNK": 3
            }
            
            # Get hidden size
            hidden_size = checkpoint.get("hidden_size", 256)
            
            # Determine model type based on state dict keys
            encoder_state_dict = checkpoint.get("encoder_state_dict") or checkpoint.get("enc_state")
            
            # Check if the encoder state dict has "lstm" in any key
            is_lstm = any("lstm" in key for key in encoder_state_dict.keys())
            model_type = "lstm" if is_lstm else "gru"
            print(f"Detected model type: {model_type}")
            
            # Initialize models using the factory function
            from seq2seq_model import create_model
            self.encoder, self.decoder = create_model(
                model_type=model_type,
                input_size=len(self.src_vocab),
                output_size=len(self.tgt_vocab),
                hidden_size=hidden_size
            )
            
            # Load state dictionaries
            self.encoder.load_state_dict(encoder_state_dict)
            
            decoder_state_dict = checkpoint.get("decoder_state_dict") or checkpoint.get("dec_state")
            self.decoder.load_state_dict(decoder_state_dict)
            
            # Set models to evaluation mode
            self.encoder.eval()
            self.decoder.eval()
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_text(self, text):
        """
        Preprocess text for translation
        
        Args:
            text: Raw text string
            
        Returns:
            processed_text: Tokenized text string
        """
        return " ".join(self.sp.encode(text, out_type=str))
    
    def postprocess_text(self, tokens):
        """
        Postprocess translation output
        
        Args:
            tokens: List of output tokens
            
        Returns:
            text: Detokenized text
        """
        return self.sp.decode(tokens)
    
    def translate_sentence(self, sentence):
        """
        Translate a single sentence
        
        Args:
            sentence: Source sentence string
            
        Returns:
            translation: Target sentence string
        """
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            try:
                # Preprocess input sentence
                processed_input = self.preprocess_text(sentence)
                
                # Convert to tensor
                input_tensor = tensor_from_sentence(
                    self.src_vocab, 
                    processed_input,
                    pad_idx=self.special_tokens["PAD"], 
                    eos_idx=self.special_tokens["EOS"], 
                    unk_idx=self.special_tokens["UNK"]
                ).unsqueeze(1).to(self.device)
                
                # Initialize encoder hidden state
                encoder_hidden = self.encoder.get_initial_hidden_state(batch_size=1)
                
                # Forward pass through encoder
                encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
                
                # Prepare decoder input (start with SOS token)
                decoder_input = torch.tensor([self.special_tokens["SOS"]], device=self.device)
                decoder_hidden = encoder_hidden
                
                # Store decoded tokens
                decoded_words = []
                
                # Forward pass through decoder
                for di in range(self.max_length):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, encoder_outputs, decoder_hidden)
                    
                    # Get the highest probability word
                    topv, topi = decoder_output.topk(1)
                    token_idx = topi.item()
                    
                    # If EOS token, stop decoding
                    if token_idx == self.special_tokens["EOS"]:
                        break
                    
                    # Add token to output list
                    if token_idx in self.tgt_index2word:
                        decoded_words.append(self.tgt_index2word[token_idx])
                    else:
                        decoded_words.append("<UNK>")
                    
                    # Update decoder input for next step
                    decoder_input = torch.tensor([token_idx], device=self.device)
                
                # Postprocess output
                translation = self.postprocess_text(decoded_words)
                
                return translation
                
            except Exception as e:
                print(f"Error translating sentence: {str(e)}")
                return "[Translation Error]"
    
    def translate_file(self, input_file, output_file):
        """
        Translate all sentences in a file
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            
        Returns:
            success: Whether translation was successful
        """
        try:
            print(f"Translating file: {input_file}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Read input sentences
            with open(input_file, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded {len(sentences)} sentences for translation")
            
            # Translate each sentence
            with open(output_file, "w", encoding="utf-8") as f_out:
                for idx, sentence in enumerate(tqdm(sentences, desc="Translating")):
                    translation = self.translate_sentence(sentence)
                    f_out.write(translation + "\n")
                    
                    # Print progress
                    if (idx + 1) % 10 == 0:
                        print(f"Translated {idx + 1}/{len(sentences)} sentences")
            
            print(f"Translation complete! Output saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error translating file: {str(e)}")
            return False


class BeamTranslator(Translator):
    """Translator that uses beam search for more accurate translations"""
    
    def __init__(self, model_path, sp_model_path, beam_size=5, device=None, max_length=512):
        """
        Initialize beam translator
        
        Args:
            model_path: Path to the trained model checkpoint
            sp_model_path: Path to the SentencePiece model
            beam_size: Beam search width
            device: Device to run translation on
            max_length: Maximum sequence length for translation
        """
        super(BeamTranslator, self).__init__(model_path, sp_model_path, device, max_length)
        self.beam_size = beam_size
    
    def translate_sentence(self, sentence):
        """
        Translate a single sentence using beam search
        
        Args:
            sentence: Source sentence string
            
        Returns:
            translation: Target sentence string
        """
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            try:
                # Preprocess input sentence
                processed_input = self.preprocess_text(sentence)
                
                # Convert to tensor
                input_tensor = tensor_from_sentence(
                    self.src_vocab, 
                    processed_input,
                    pad_idx=self.special_tokens["PAD"], 
                    eos_idx=self.special_tokens["EOS"], 
                    unk_idx=self.special_tokens["UNK"]
                ).unsqueeze(1).to(self.device)
                
                # Initialize encoder hidden state
                encoder_hidden = self.encoder.get_initial_hidden_state(batch_size=1)
                
                # Forward pass through encoder
                encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
                
                # Initialize beam search
                # Each beam contains: (cumulative_score, decoded_sequence, current_decoder_input, current_decoder_hidden)
                beams = [(0.0, [self.special_tokens["SOS"]], 
                         torch.tensor([self.special_tokens["SOS"]], device=self.device), 
                         encoder_hidden)]
                finished_beams = []
                
                # Beam search
                for _ in range(self.max_length - 1):
                    candidates = []
                    
                    for score, sequence, decoder_input, decoder_hidden in beams:
                        # If the last token is EOS, add the beam to finished beams
                        if sequence[-1] == self.special_tokens["EOS"]:
                            finished_beams.append((score, sequence))
                            continue
                        
                        # Forward pass through decoder
                        decoder_output, new_decoder_hidden = self.decoder(
                            decoder_input, encoder_outputs, decoder_hidden)
                        
                        # Get top k tokens
                        log_probs, indices = decoder_output.topk(self.beam_size)
                        
                        # Add candidates
                        for i in range(self.beam_size):
                            token_idx = indices[0][i].item()
                            token_log_prob = log_probs[0][i].item()
                            
                            # Create new beam
                            new_score = score + token_log_prob
                            new_sequence = sequence + [token_idx]
                            new_decoder_input = torch.tensor([token_idx], device=self.device)
                            
                            candidates.append((new_score, new_sequence, new_decoder_input, new_decoder_hidden))
                    
                    # If no candidates (all beams ended with EOS), break
                    if not candidates:
                        break
                    
                    # Sort candidates by score (descending) and keep top beam_size
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    beams = candidates[:self.beam_size]
                
                # Add any remaining beams to finished beams
                for score, sequence, _, _ in beams:
                    if sequence[-1] != self.special_tokens["EOS"]:
                        sequence.append(self.special_tokens["EOS"])
                    finished_beams.append((score, sequence))
                
                # Sort finished beams by score and get the best one
                if finished_beams:
                    finished_beams.sort(key=lambda x: x[0], reverse=True)
                    best_sequence = finished_beams[0][1]
                else:
                    best_sequence = [self.special_tokens["SOS"], self.special_tokens["EOS"]]
                
                # Convert token indices to words (excluding SOS and EOS)
                decoded_words = []
                for token_idx in best_sequence[1:-1]:  # Skip SOS and EOS
                    if token_idx in self.tgt_index2word:
                        decoded_words.append(self.tgt_index2word[token_idx])
                    else:
                        decoded_words.append("<UNK>")
                
                # Postprocess output
                translation = self.postprocess_text(decoded_words)
                
                return translation
                
            except Exception as e:
                print(f"Error translating sentence with beam search: {str(e)}")
                return "[Translation Error]"


def main():
    """Main function to run translation"""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Translate text using a trained neural machine translation model')
    parser.add_argument('--model', default="models/model_checkpoint.pt", help='Path to the model checkpoint')
    parser.add_argument('--sp-model', default="models/spm_ru_en.model", help='Path to SentencePiece model')
    parser.add_argument('--input', default="data/test_preprocessed_ru.txt", help='Input file with Russian text')
    parser.add_argument('--output', default="data/wmt_test_translations.txt", help='Output file for translations')
    parser.add_argument('--beam-size', type=int, default=0, help='Beam size for beam search (0 for greedy)')
    args = parser.parse_args()
    
    # Determine whether to use beam search
    use_beam_search = args.beam_size > 0
    
    # Create translator
    if use_beam_search:
        translator = BeamTranslator(args.model, args.sp_model, beam_size=args.beam_size)
    else:
        translator = Translator(args.model, args.sp_model)
    
    # Translate file
    translator.translate_file(args.input, args.output)


if __name__ == "__main__":
    main()