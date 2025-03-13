import torch
import random
import time
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from seq2seq_attention import EncoderRNN, AttnDecoderRNN, tensors_from_pair

# Paths
train_file = "bpe_parallel_corpus.tsv"
checkpoint_file = "model_checkpoint.pt"

# Training settings
hidden_size = 256
num_epochs = 10  # Increase if training longer
learning_rate = 0.001
batch_size = 32
teacher_forcing_ratio = 0.5
max_length = 512  # Max sequence length

# Load dataset
print("Loading training data...")
df = pd.read_csv(train_file, sep="\t", encoding="utf-8").dropna()

# Create vocab mappings
src_vocab = {word: i for i, word in enumerate(set(" ".join(df["English"]).split()))}
tgt_vocab = {word: i for i, word in enumerate(set(" ".join(df["Russian"]).split()))}

# Reverse mappings
src_index2word = {i: word for word, i in src_vocab.items()}
tgt_index2word = {i: word for word, i in tgt_vocab.items()}

print(f"Vocab size: EN = {len(src_vocab)}, RU = {len(tgt_vocab)}")

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderRNN(len(src_vocab), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, len(tgt_vocab), attn_type="dot", num_head=1).to(device)

# Optimizer and loss
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
criterion = nn.NLLLoss(ignore_index=0)  # Ignore padding

# Function to get batches
def get_batches(data, batch_size):
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        src_tensors, tgt_tensors = zip(*[tensors_from_pair(src_vocab, tgt_vocab, (row["English"], row["Russian"])) for _, row in batch.iterrows()])
        yield torch.cat(src_tensors, dim=1), torch.cat(tgt_tensors, dim=1)

# Training function
def train_epoch(encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()
    total_loss = 0
    for src_batch, tgt_batch in get_batches(df, batch_size):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        
        optimizer.zero_grad()
        loss = 0
        encoder_hidden = encoder.get_initial_hidden_state()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(src_batch.size(0)):
            encoder_output, encoder_hidden = encoder(src_batch[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        
        encoder_outputs = encoder_outputs.unsqueeze(0)
        
        decoder_input = torch.tensor([[0]] * batch_size, device=device)  # SOS tokens
        decoder_hidden = encoder_hidden

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        for di in range(tgt_batch.size(0)):
            decoder_output, decoder_hidden = decoder(decoder_input, encoder_outputs, decoder_hidden)
            loss += criterion(decoder_output, tgt_batch[di])

            if use_teacher_forcing:
                decoder_input = tgt_batch[di]
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() / tgt_batch.size(0)
    
    return total_loss / len(df)

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

print("✅ Training complete!")

