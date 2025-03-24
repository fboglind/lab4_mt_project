"""seq2seq_model.py
Model definitions for the sequence-to-sequence architecture with attention.
Supports both GRU and LSTM encoders/decoders, attention mechanisms,
and implements tensor conversion for training and inference.
"""


import torch
from torch import nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """Abstract base class for encoder models"""

    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(BaseEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Common layers for all encoders
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_seq, hidden):
        """Forward pass through the encoder
        Args:
            input_seq: Input sequence tensor [seq_len, batch_size]
            hidden: Initial hidden state

        Returns:
            outputs: Outputs from the encoder [seq_len, batch_size, hidden_size]
            hidden: Final hidden state
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_initial_hidden_state(self, batch_size=1):
        """Initialize hidden state with zeros"""
        raise NotImplementedError(
            "Subclasses must implement get_initial_hidden_state method"
        )


class GRUEncoder(BaseEncoder):
    """GRU-based encoder implementation"""

    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(GRUEncoder, self).__init__(input_size, hidden_size, n_layers, dropout_p)

        # GRU layer
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout_p if n_layers > 1 else 0,
            batch_first=False,  # [seq_len, batch, hidden]
        )

    def forward(self, input_seq, hidden):
        """Forward pass through the GRU encoder
        Args:
            input_seq: Input sequence tensor [seq_len, batch_size]
            hidden: Initial hidden state [n_layers, batch_size, hidden_size]

        Returns:
            outputs: Outputs from the GRU [seq_len, batch_size, hidden_size]
            hidden: Final hidden state [n_layers, batch_size, hidden_size]
        """
        # Create embeddings [seq_len, batch_size, hidden_size]
        embedded = self.dropout(self.embedding(input_seq))

        # Pass through GRU
        outputs, hidden = self.gru(embedded, hidden)

        return outputs, hidden

    def get_initial_hidden_state(self, batch_size=1):
        """Initialize hidden state with zeros"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)


class LSTMEncoder(BaseEncoder):
    """LSTM-based encoder implementation"""

    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(LSTMEncoder, self).__init__(input_size, hidden_size, n_layers, dropout_p)

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout_p if n_layers > 1 else 0,
            batch_first=False,  # [seq_len, batch, hidden]
        )

    def forward(self, input_seq, hidden):
        """
        Forward pass through the LSTM encoder
        Args:
            input_seq: Input sequence tensor [seq_len, batch_size]
            hidden: Tuple of initial hidden state and cell state
                   (h_0, c_0) where each has shape [n_layers, batch_size, hidden_size]

        Returns:
            outputs: Outputs from the LSTM [seq_len, batch_size, hidden_size]
            hidden: Tuple of final hidden state and cell state
        """
        # Create embeddings [seq_len, batch_size, hidden_size]
        embedded = self.dropout(self.embedding(input_seq))

        # Pass through LSTM
        outputs, hidden = self.lstm(embedded, hidden)

        return outputs, hidden

    def get_initial_hidden_state(self, batch_size=1):
        """Initialize hidden state and cell state with zeros
        Args:
            batch_size: Size of the batch
        Returns:
            hidden: Tuple of (h_0, c_0) each with shape [n_layers, batch_size, hidden_size]
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


class BaseAttention(nn.Module):
    """Abstract base class for attention mechanisms"""

    def __init__(self, hidden_size, method="general"):
        super(BaseAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method

    def forward(self, decoder_hidden, encoder_outputs):
        """Calculate attention weights
        Args:
            decoder_hidden: Current decoder hidden state
            encoder_outputs: All encoder outputs

        Returns:
            attention_weights: Attention weight distribution over encoder outputs
        """
        raise NotImplementedError("Subclasses must implement forward method")


class AdditiveAttention(BaseAttention):
    """Bahdanau (additive) attention implementation"""

    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__(hidden_size, method="additive")

        # Attention layers
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Calculate attention weights using additive attention
        Args:
            decoder_hidden: Current decoder hidden state [1, batch_size, hidden_size]
            encoder_outputs: All encoder outputs [seq_len, batch_size, hidden_size]

        Returns:
            attention_weights: Attention weight distribution [batch_size, seq_len]
        """
        # Get dimensions
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Expand decoder hidden state to match encoder outputs length
        decoder_hidden = decoder_hidden[0].unsqueeze(1)  # [batch_size, 1, hidden_size]
        decoder_hidden = decoder_hidden.repeat(
            1, seq_len, 1
        )  # [batch_size, seq_len, hidden_size]

        # Transpose encoder outputs to match batch-first convention
        encoder_outputs = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]

        # Concatenate decoder hidden state with each encoder output
        attn_inputs = torch.cat(
            (decoder_hidden, encoder_outputs), dim=2
        )  # [batch_size, seq_len, hidden_size*2]

        # Calculate attention energies
        energy = torch.tanh(
            self.attn(attn_inputs)
        )  # [batch_size, seq_len, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        # Normalize with softmax
        attention_weights = F.softmax(attention, dim=1)  # [batch_size, seq_len]

        return attention_weights


class BaseDecoder(nn.Module):
    """Abstract base class for decoder models"""

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(BaseDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        # Common layers for all decoders
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, encoder_outputs, hidden):
        """Forward pass through the decoder
        Args:
            input: Input token indices [batch_size]
            encoder_outputs: All outputs from the encoder
            hidden: Current hidden state

        Returns:
            output: Output token probabilities [batch_size, output_size]
            hidden: Updated hidden state
        """
        raise NotImplementedError("Subclasses must implement forward method")


class GRUAttnDecoder(BaseDecoder):
    """GRU decoder with attention"""

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(GRUAttnDecoder, self).__init__(hidden_size, output_size, dropout_p)

        # Create attention mechanism
        self.attention = AdditiveAttention(hidden_size)

        # Layers specific to GRU decoder with attention
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, encoder_outputs, hidden):
        """Forward pass through the GRU decoder with attention
        Args:
            input: Input token indices [batch_size]
            encoder_outputs: All outputs from the encoder [seq_len, batch_size, hidden_size]
            hidden: Current hidden state [1, batch_size, hidden_size]

        Returns:
            output: Output token probabilities [batch_size, output_size]
            hidden: Updated hidden state [1, batch_size, hidden_size]
        """
        # Embedding layer
        embedded = self.dropout(self.embedding(input))  # [batch_size, hidden_size]

        # Calculate attention weights
        attn_weights = self.attention(hidden, encoder_outputs)  # [batch_size, seq_len]

        # Apply attention weights to encoder outputs
        encoder_outputs_transposed = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            encoder_outputs_transposed,  # [batch_size, seq_len, hidden_size]
        )  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]

        # Combine embedding and context
        rnn_input = torch.cat((embedded, context), dim=1)  # [batch_size, hidden_size*2]
        rnn_input = self.attn_combine(rnn_input)  # [batch_size, hidden_size]
        rnn_input = F.relu(rnn_input)

        # Reshape for GRU input
        rnn_input = rnn_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        # GRU forward pass
        output, hidden = self.gru(rnn_input, hidden)

        # Project to vocabulary size
        output = self.softmax(self.out(output[0]))  # [batch_size, output_size]

        return output, hidden


class LSTMAttnDecoder(BaseDecoder):
    """LSTM decoder with attention"""

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(LSTMAttnDecoder, self).__init__(hidden_size, output_size, dropout_p)

        # Create attention mechanism
        self.attention = AdditiveAttention(hidden_size)

        # Layers specific to LSTM decoder with attention
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, encoder_outputs, hidden):
        """Forward pass through the LSTM decoder with attention
        Args:
            input: Input token indices [batch_size]
            encoder_outputs: All outputs from the encoder [seq_len, batch_size, hidden_size]
            hidden: Tuple of current hidden state and cell state
                   (h_n, c_n) where each has shape [1, batch_size, hidden_size]

        Returns:
            output: Output token probabilities [batch_size, output_size]
            hidden: Updated tuple of hidden state and cell state
        """
        # Embedding layer
        embedded = self.dropout(self.embedding(input))  # [batch_size, hidden_size]

        # Calculate attention weights - using only the hidden state (h_n), not the cell state (c_n)
        attn_weights = self.attention(
            hidden[0], encoder_outputs
        )  # [batch_size, seq_len]

        # Apply attention weights to encoder outputs
        encoder_outputs_transposed = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            encoder_outputs_transposed,  # [batch_size, seq_len, hidden_size]
        )  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]

        # Combine embedding and context
        rnn_input = torch.cat((embedded, context), dim=1)  # [batch_size, hidden_size*2]
        rnn_input = self.attn_combine(rnn_input)  # [batch_size, hidden_size]
        rnn_input = F.relu(rnn_input)

        # Reshape for LSTM input
        rnn_input = rnn_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        # LSTM forward pass
        output, hidden = self.lstm(rnn_input, hidden)

        # Project to vocabulary size
        output = self.softmax(self.out(output[0]))  # [batch_size, output_size]

        return output, hidden


# Helper functions for data processing
def tensor_from_sentence(
    vocab, sentence, pad_idx=0, eos_idx=2, unk_idx=3, max_length=512
):
    """Convert a sentence to a tensor of indices, adding EOS token at the end

    Args:
        vocab: Dictionary mapping words to indices
        sentence: String containing space-separated tokens
        pad_idx: Index of padding token
        eos_idx: Index of end-of-sentence token
        unk_idx: Index of unknown token
        max_length: Maximum sequence length

    Returns:
        tensor: Tensor of token indices
    """
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


def tensors_from_pair(
    src_vocab, tgt_vocab, pair, pad_idx=0, eos_idx=2, unk_idx=3, max_length=512
):
    """Convert a sentence pair to tensors

    Args:
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        pair: Tuple of (source_sentence, target_sentence)
        pad_idx, eos_idx, unk_idx, max_length: Parameters for tensor_from_sentence

    Returns:
        src_tensor, tgt_tensor: Tensors of token indices
    """
    src_tensor = tensor_from_sentence(
        src_vocab, pair[0], pad_idx, eos_idx, unk_idx, max_length
    )
    tgt_tensor = tensor_from_sentence(
        tgt_vocab, pair[1], pad_idx, eos_idx, unk_idx, max_length
    )
    return src_tensor, tgt_tensor


def create_model(
    model_type, input_size, output_size, hidden_size=256, n_layers=1, dropout_p=0.1
):
    """Creates encoder and decoder based on model type

    Args:
        model_type: Type of model ("gru" or "lstm")
        input_size: Size of input vocabulary
        output_size: Size of output vocabulary
        hidden_size: Size of hidden layers
        n_layers: Number of layers in encoder/decoder
        dropout_p: Dropout probability

    Returns:
        encoder, decoder: Encoder and decoder models
    """
    if model_type.lower() == "gru":
        encoder = GRUEncoder(input_size, hidden_size, n_layers, dropout_p)
        decoder = GRUAttnDecoder(hidden_size, output_size, dropout_p)
    elif model_type.lower() == "lstm":
        encoder = LSTMEncoder(input_size, hidden_size, n_layers, dropout_p)
        decoder = LSTMAttnDecoder(hidden_size, output_size, dropout_p)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gru' or 'lstm'.")

    return encoder, decoder
