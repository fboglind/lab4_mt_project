"""seq2seq_model.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Embedding layer: convert token indices to vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
                          dropout=dropout_p if n_layers > 1 else 0,
                          batch_first=False)  # batch_first=False means [seq_len, batch, hidden]

    def forward(self, input, hidden):
        # input shape: [seq_len, batch_size]
        
        # Create embeddings: [seq_len, batch_size, hidden_size]
        embedded = self.dropout(self.embedding(input))
        
        # Pass through GRU
        # outputs: [seq_len, batch_size, hidden_size]
        # hidden: [n_layers, batch_size, hidden_size]
        outputs, hidden = self.gru(embedded, hidden)
        
        return outputs, hidden

    def get_initial_hidden_state(self, batch_size=1):
        """Initialize hidden state with zeros"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=512):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Layers for attention mechanism
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        
        # Attention layers - using a different approach that doesn't rely on fixed sequence length
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, 1, bias=False)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        
        # GRU and output layers
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # LogSoftmax for output probabilities
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, encoder_outputs, hidden):
        """
        input: [batch_size]
        encoder_outputs: [seq_len, batch_size, hidden_size]
        hidden: [1, batch_size, hidden_size]
        """
        # Get dimensions
        seq_len = encoder_outputs.size(0)
        batch_size = input.size(0)
        
        # Embedding
        embedded = self.dropout(self.embedding(input))  # [batch_size, hidden_size]
        
        # Calculate attention weights - approach works with any sequence length
        # Repeat hidden state seq_len times
        hidden_expanded = hidden[0].unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]
        
        # Transpose encoder outputs to match batch-first convention
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # Concatenate hidden state with each encoder output
        attn_inputs = torch.cat((hidden_expanded, encoder_outputs), dim=2)  # [batch_size, seq_len, hidden_size*2]
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(attn_inputs))  # [batch_size, seq_len, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attention, dim=1)  # [batch_size, seq_len]
        
        # Apply attention weights to encoder outputs
        # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_size] -> [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        # Combine embedded input and attention context
        output = torch.cat((embedded, context), 1)  # [batch_size, hidden_size*2]
        output = self.attn_combine(output)  # [batch_size, hidden_size]
        
        # Add non-linearity
        output = F.relu(output)
        
        # Reshape for GRU input: [1, batch_size, hidden_size]
        output = output.unsqueeze(0)
        
        # Pass through GRU
        output, hidden = self.gru(output, hidden)
        
        # Final output probability distribution
        output = self.softmax(self.out(output[0]))  # [batch_size, output_size]
        
        return output, hidden

def tensor_from_sentence(vocab, sentence, pad_idx=0, eos_idx=2, unk_idx=3, max_length=512):
    """Convert a sentence to a tensor of indices, adding EOS token at the end"""
    
    # Split the sentence into tokens
    words = sentence.split()
    
    # Convert tokens to indices, handling unknown words
    indices = []
    for word in words:
        if word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(unk_idx)  # UNK token
    
    # Add EOS token
    indices.append(eos_idx)
    
    # Truncate if needed
    indices = indices[:max_length]
    
    return torch.tensor(indices, dtype=torch.long)

def tensors_from_pair(src_vocab, tgt_vocab, pair, pad_idx=0, eos_idx=2, unk_idx=3, max_length=512):
    """Convert a sentence pair to tensors"""
    src_tensor = tensor_from_sentence(src_vocab, pair[0], pad_idx, eos_idx, unk_idx, max_length)
    tgt_tensor = tensor_from_sentence(tgt_vocab, pair[1], pad_idx, eos_idx, unk_idx, max_length)
    return src_tensor, tgt_tensor