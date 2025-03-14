"""seq2seq_train.py"""
import time
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
from seq2seq_model import EncoderRNN, AttnDecoderRNN, tensor_from_sentence, tensors_from_pair

# Paths
train_file = "data/spm_parallel_corpus.tsv"
checkpoint_file = "models/model_checkpoint.pt"

# Training settings
hidden_size = 256
num_epochs = 15
learning_rate = 0.0005
batch_size = 32
teacher_forcing_ratio = 0.5
max_length = 512  # Long enough to handle abstracts
dropout_p = 0.1

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Load dataset
print("Loading training data...")
df = pd.read_csv(train_file, sep="\t", encoding="utf-8").dropna()
print(f"Number of samples after dropping NaN: {len(df)}")

# Create vocabularies with special tokens
src_vocab = {"<pad>": PAD_token, "<sos>": SOS_token, "<eos>": EOS_token, "<unk>": UNK_token}
tgt_vocab = {"<pad>": PAD_token, "<sos>": SOS_token, "<eos>": EOS_token, "<unk>": UNK_token}

# Add words from dataset
print("Building vocabularies...")
for i, word in enumerate(set(" ".join(df["Russian"]).split())):
    if word not in src_vocab:
        src_vocab[word] = len(src_vocab)
        
for i, word in enumerate(set(" ".join(df["English"]).split())):
    if word not in tgt_vocab:
        tgt_vocab[word] = len(tgt_vocab)

# Create reverse mappings for decoding
src_index2word = {idx: word for word, idx in src_vocab.items()}
tgt_index2word = {idx: word for word, idx in tgt_vocab.items()}

print(f"Vocabulary sizes: Russian = {len(src_vocab)}, English = {len(tgt_vocab)}")

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

encoder = EncoderRNN(len(src_vocab), hidden_size, n_layers=1, dropout_p=dropout_p).to(device)
decoder = AttnDecoderRNN(hidden_size, len(tgt_vocab), dropout_p=dropout_p, max_length=max_length).to(device)

# Optimizer and loss
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=learning_rate
)
criterion = nn.NLLLoss(ignore_index=PAD_token)

def get_batches(data, batch_size, pad_idx=PAD_token):
    """Shuffle and return batches of (source, target) tensors with padding"""
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:min(i + batch_size, len(data))]  # Handle last batch
        
        src_tensors = []
        tgt_tensors = []
        
        for _, row in batch.iterrows():
            src_tensor = tensor_from_sentence(
                src_vocab, row["Russian"], 
                pad_idx=PAD_token, eos_idx=EOS_token, unk_idx=UNK_token
            )
            tgt_tensor = tensor_from_sentence(
                tgt_vocab, row["English"], 
                pad_idx=PAD_token, eos_idx=EOS_token, unk_idx=UNK_token
            )
            src_tensors.append(src_tensor)
            tgt_tensors.append(tgt_tensor)
        
        # Ensure all tensors in the batch have the same length
        src_max_len = max([tensor.size(0) for tensor in src_tensors])
        tgt_max_len = max([tensor.size(0) for tensor in tgt_tensors])
        
        # Pad to max length in batch
        for i in range(len(src_tensors)):
            src_padding = torch.full((src_max_len - src_tensors[i].size(0),), PAD_token, dtype=torch.long)
            tgt_padding = torch.full((tgt_max_len - tgt_tensors[i].size(0),), PAD_token, dtype=torch.long)
            src_tensors[i] = torch.cat([src_tensors[i], src_padding])
            tgt_tensors[i] = torch.cat([tgt_tensors[i], tgt_padding])
        
        # Stack tensors into a batch [batch_size, seq_len]
        src_batch = torch.stack(src_tensors)
        tgt_batch = torch.stack(tgt_tensors)
        
        # Transpose to [seq_len, batch_size] for RNN processing
        src_batch = src_batch.transpose(0, 1)
        tgt_batch = tgt_batch.transpose(0, 1)

        yield src_batch, tgt_batch

def train_epoch(encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()
    total_loss = 0
    batch_count = 0
    total_samples = 0

    # Process batches
    for src_batch, tgt_batch in get_batches(df, batch_size):
        batch_size_current = src_batch.size(1)  # Get batch size outside try block
        
        try:
            # Move tensors to device
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            loss = 0
            
            # Initialize encoder hidden state
            encoder_hidden = encoder.get_initial_hidden_state(batch_size_current)
            
            # Forward pass through encoder
            encoder_outputs, encoder_hidden = encoder(src_batch, encoder_hidden)
            
            # Prepare decoder input (start with SOS tokens)
            decoder_input = torch.full((batch_size_current,), SOS_token, dtype=torch.long, device=device)
            decoder_hidden = encoder_hidden
            
            # Teacher forcing: use target as the next input
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            # Forward pass through decoder
            target_length = tgt_batch.size(0)
            max_output_length = min(target_length, max_length)
            
            # Debug checks to ensure no NaN values
            if torch.isnan(encoder_outputs).any():
                print("⚠️ NaN detected in encoder outputs!")
                continue
                
            if torch.isnan(encoder_hidden).any():
                print("⚠️ NaN detected in encoder hidden state!")
                continue
            
            # Decode one step at a time
            for t in range(1, max_output_length):  # Start from 1 to skip the SOS token
                try:
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, encoder_outputs, decoder_hidden)
                    
                    # Check for NaNs in decoder output
                    if torch.isnan(decoder_output).any():
                        print(f"⚠️ NaN detected in decoder output at step {t}!")
                        break
                    
                    if torch.isnan(decoder_hidden).any():
                        print(f"⚠️ NaN detected in decoder hidden state at step {t}!")
                        break
                    
                    # Compute loss for this step
                    step_loss = criterion(decoder_output, tgt_batch[t])
                    loss += step_loss
                    
                    # Determine next input
                    if use_teacher_forcing:
                        decoder_input = tgt_batch[t]  # Teacher forcing: use ground truth
                    else:
                        # Without teacher forcing: use decoder's own prediction
                        _, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze(-1).detach()  # Detach from history as input
                        
                except Exception as e:
                    print(f"Error in decoding step {t} of batch {batch_count}: {str(e)}")
                    continue
            
            # Check if loss is NaN before backprop
            if torch.isnan(loss):
                print(f"⚠️ NaN detected in loss at batch {batch_count}!")
                continue
            
            # Normalize loss by sequence length to stabilize training
            normalized_loss = loss / (max_output_length - 1)  # -1 because we skip the first token
            
            # Backpropagation
            normalized_loss.backward()
            
            # Gradient clipping to prevent exploding gradients (reduced max_norm for stability)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=0.5)
            
            # Update parameters
            optimizer.step()
            
            # Track loss
            batch_loss = normalized_loss.item()
            total_loss += batch_loss
            batch_count += 1
            total_samples += batch_size_current
            
            # Print batch progress
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count} - Loss: {batch_loss:.4f}")
        
        except Exception as e:
            print(f"Error in batch {batch_count}: {str(e)}")
            continue
    
    # Return average loss
    return total_loss / batch_count if batch_count > 0 else float('inf')

# Training loop
print("Starting training...")
best_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    
    try:
        # Train for one epoch
        loss = train_epoch(encoder, decoder, optimizer, criterion)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Save model checkpoint
        if loss < best_loss:
            best_loss = loss
            checkpoint = {
                "epoch": epoch,
                "hidden_size": hidden_size,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "src_index2word": src_index2word,
                "tgt_index2word": tgt_index2word,
                "enc_state": encoder.state_dict(),
                "dec_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss
            }
            
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved: {checkpoint_file}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        break
    
    except Exception as e:
        print(f"Error in epoch {epoch}: {str(e)}")
        continue

print("Training complete!")