"""seq2seq_model.py"""
import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        # input shape: [seq_len, batch_size]
        embedded = self.embedding(input)  # [seq_len, batch_size, hidden_size]
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    # def get_initial_hidden_state(self, batch_size=1):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    # In seq2seq_model.py, update the get_initial_hidden_state method:

    def get_initial_hidden_state(self, batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_type="dot", num_head=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, encoder_outputs, hidden):
        # input shape: [batch_size]
        batch_size = input.size(0)
        input = input.unsqueeze(0)

        # Embedding: [1, batch_size] -> [1, batch_size, hidden_size]
        embedded = self.embedding(input)
        # embedded = self.embedding(input).view(1, 1, -1)

        output, hidden = self.gru(embedded, hidden)
    
        # Output: [1, batch_size, hidden_size] -> [batch_size, output_size]
        output = self.softmax(self.out(output[0]))

        #output, hidden = self.gru(embedded, hidden)
        #output = self.softmax(self.out(output[0]))
        return output, hidden

def tensor_from_sentence(vocab, sentence, pad_idx=0, max_length=512):
    indices = [vocab[word] for word in sentence.split() if word in vocab]
    
    # Pad sequence to max_length
    indices = indices[:max_length]  # Truncate if too long
    while len(indices) < max_length:
        indices.append(pad_idx)
    
    return torch.tensor(indices, dtype=torch.long)  # Removed .view(-1, 1)

# Helper function to convert a parallel sentence pair into tensors
def tensors_from_pair(src_vocab, tgt_vocab, pair):
    return tensor_from_sentence(src_vocab, pair[0]), tensor_from_sentence(tgt_vocab, pair[1])
