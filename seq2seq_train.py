"""seq2seq_train.py"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pandas as pd
from seq2seq_model import EncoderRNN, AttnDecoderRNN, tensors_from_pair

# Paths
train_file = "bpe_parallel_corpus.tsv"
checkpoint_file = "model_checkpoint.pt"

# Training settings
hidden_size = 256
num_epochs = 15
learning_rate = 0.0005
batch_size = 32
teacher_forcing_ratio = 0.35
max_length = 512

# Load dataset
print("Loading training data...")
df = pd.read_csv("spm_parallel_corpus.tsv", sep="\t", encoding="utf-8").dropna()

# Define source and target: Russian → English
src_vocab = {word: i for i, word in enumerate(set(" ".join(df["Russian"]).split()))}
tgt_vocab = {word: i for i, word in enumerate(set(" ".join(df["English"]).split()))}

print(f"Vocab size: RU = {len(src_vocab)}, EN = {len(tgt_vocab)}")

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
encoder = EncoderRNN(len(src_vocab), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(tgt_vocab)).to(device)

# Optimizer and loss
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
criterion = nn.NLLLoss(ignore_index=0)

def get_batches(data, batch_size, pad_idx=0):
    """ Shuffle and return batches of (source, target) tensors with padding """
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

    for i in range(0, len(data), batch_size):
        
        batch = data.iloc[i:min(i + batch_size, len(data))]  # Handle last batch
        
        src_tensors, tgt_tensors = zip(*[tensors_from_pair(src_vocab, tgt_vocab, (row["English"], row["Russian"])) 
                                       for _, row in batch.iterrows()])
        
        # Stack tensors into a batch
        src_batch = torch.stack(src_tensors, dim=1)  # [seq_len, batch_size]
        tgt_batch = torch.stack(tgt_tensors, dim=1)  # [seq_len, batch_size]

        yield src_batch, tgt_batch

# Training function
def train_epoch(encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()
    total_loss = 0
    batch_count = 0
    
    for src_batch, tgt_batch in get_batches(df, batch_size):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        
        # Get actual batch size from current batch
        current_batch_size = src_batch.size(1)  # [seq_len, batch_size]
        
        optimizer.zero_grad()  # Reset gradients
        
        loss = 0
        encoder_hidden = encoder.get_initial_hidden_state(current_batch_size)
        
        # Create encoder_outputs with the correct batch size
        encoder_outputs = torch.zeros(max_length, current_batch_size, encoder.hidden_size, device=device)
        
        # Pass data through the encoder
        embedded = encoder.embedding(src_batch)
        encoder_output, encoder_hidden = encoder.gru(embedded, encoder_hidden)
        
        # Store encoder outputs (using the correct size)
        encoder_outputs[:encoder_output.size(0)] = encoder_output
        
        # Continue with the rest of your training code...
        # (Using current_batch_size instead of batch_size)
        decoder_input = torch.full((current_batch_size,), 0, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
    
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        for di in range(tgt_batch.size(0)):
            #decoder_output, decoder_hidden = decoder(decoder_input.squeeze(1), encoder_outputs, decoder_hidden)
            decoder_output, decoder_hidden = decoder(decoder_input, encoder_outputs, decoder_hidden)
            loss += criterion(decoder_output, tgt_batch[di])

            if use_teacher_forcing:
                decoder_input = tgt_batch[di]  # Already has shape [batch_size]
                #decoder_input = tgt_batch[di].unsqueeze(1)  # Ensure correct shape for next step
            else:
                topv, topi = decoder_output.topk(1)
                #decoder_input = topi.squeeze().detach().unsqueeze(1)
                decoder_input = topi.squeeze(-1).detach()  # Ensure shape is [batch_size]

        loss.backward()  # Backpropagation

        # Apply gradient clipping - helps prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)

        optimizer.step()  # Apply gradients

        total_loss += loss.item() / tgt_batch.size(0)

        batch_count += 1

    return total_loss / batch_count  # Average loss per batch

# Training loop
print("Starting training...")
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    loss = train_epoch(encoder, decoder, optimizer, criterion)
    print(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} - Time: {time.time() - start_time:.2f}s")
    
    # Save model checkpoint
    checkpoint = {
        "hidden_size": hidden_size,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "enc_state": encoder.state_dict(),
        "dec_state": decoder.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved: {checkpoint_file}")

print("Training complete!")
