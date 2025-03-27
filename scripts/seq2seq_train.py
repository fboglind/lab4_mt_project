"""Training script for a sequence-to-sequence (seq2seq) machine translation model 
with attention. Supports GRU or LSTM, gradient accumulation, and teacher forcing.
"""

import os
import time
import random
import argparse
import pandas as pd
import torch
from torch import nn, optim
from seq2seq_model import create_model, tensor_from_sentence


# Special token indices
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class DataLoader:
    """Class for loading and processing training data"""

    def __init__(
        self,
        data_path,
        batch_size,
        src_lang_col,
        tgt_lang_col,
        special_tokens=None,
        max_vocab_size=30000,
    ):
        """
        Initialize data loader

        Args:
            data_path: Path to the parallel corpus file
            batch_size: Batch size for training
            src_lang_col: Column name for source language
            tgt_lang_col: Column name for target language
            special_tokens: Dictionary of special tokens {name: index}
            max_vocab_size: Maximum vocabulary size for each language
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.src_lang_col = src_lang_col
        self.tgt_lang_col = tgt_lang_col
        self.max_vocab_size = max_vocab_size

        # Special tokens
        self.special_tokens = special_tokens or {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}

        # Vocabularies
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_index2word = None
        self.tgt_index2word = None

        # Data
        self.df = None
        self.load_data()

    def __len__(self):
        """Return number of batches per epoch"""
        return (len(self.df) + self.batch_size - 1) // self.batch_size


    def load_data(self):
        """Load and preprocess the dataset"""
        print(f"Loading training data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, sep="\t", encoding="utf-8").dropna()
        print(f"Loaded {len(self.df)} samples after dropping NaN values")

        # Build vocabularies
        self.build_vocabularies()

    def build_vocabularies(self, max_vocab_size=30000):
        """Build source and target vocabularies from the dataset with a maximum size limit"""
        print("Building vocabularies...")

        # Initialize vocabularies with special tokens
        self.src_vocab = {
            "<pad>": self.special_tokens["PAD"],
            "<sos>": self.special_tokens["SOS"],
            "<eos>": self.special_tokens["EOS"],
            "<unk>": self.special_tokens["UNK"],
        }

        self.tgt_vocab = {
            "<pad>": self.special_tokens["PAD"],
            "<sos>": self.special_tokens["SOS"],
            "<eos>": self.special_tokens["EOS"],
            "<unk>": self.special_tokens["UNK"],
        }

        # Count token frequencies
        src_token_counts = {}
        tgt_token_counts = {}

        # Process source language tokens
        for text in self.df[self.src_lang_col]:
            for word in text.split():
                src_token_counts[word] = src_token_counts.get(word, 0) + 1

        # Process target language tokens
        for text in self.df[self.tgt_lang_col]:
            for word in text.split():
                tgt_token_counts[word] = tgt_token_counts.get(word, 0) + 1

        # Sort tokens by frequency (most frequent first)
        src_tokens = sorted(src_token_counts.items(), key=lambda x: x[1], reverse=True)
        tgt_tokens = sorted(tgt_token_counts.items(), key=lambda x: x[1], reverse=True)

        # Add most frequent tokens to vocabularies (up to max_vocab_size)
        for word, _ in src_tokens[: max_vocab_size - len(self.src_vocab)]:
            if word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)

        for word, _ in tgt_tokens[: max_vocab_size - len(self.tgt_vocab)]:
            if word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)

        # Create reverse mappings
        self.src_index2word = {idx: word for word, idx in self.src_vocab.items()}
        self.tgt_index2word = {idx: word for word, idx in self.tgt_vocab.items()}

        print(
            f"Vocabulary sizes: Source = {len(self.src_vocab)}, Target = {len(self.tgt_vocab)}"
        )

    def get_batches(self, max_length=512):
        """
        Generate batches of data for training

        Args:
            max_length: Maximum sequence length

        Returns:
            Generator yielding (src_batch, tgt_batch) tuples
        """
        # Shuffle data
        data = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle data

        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[
                i : min(i + self.batch_size, len(data))
            ]  # Handle last batch

            src_tensors = []
            tgt_tensors = []

            for _, row in batch.iterrows():
                src_tensor = tensor_from_sentence(
                    self.src_vocab,
                    row[self.src_lang_col],
                    pad_idx=self.special_tokens["PAD"],
                    eos_idx=self.special_tokens["EOS"],
                    unk_idx=self.special_tokens["UNK"],
                    max_length=max_length,
                )
                tgt_tensor = tensor_from_sentence(
                    self.tgt_vocab,
                    row[self.tgt_lang_col],
                    pad_idx=self.special_tokens["PAD"],
                    eos_idx=self.special_tokens["EOS"],
                    unk_idx=self.special_tokens["UNK"],
                    max_length=max_length,
                )
                src_tensors.append(src_tensor)
                tgt_tensors.append(tgt_tensor)

            # Ensure all tensors in the batch have the same length
            src_max_len = max([tensor.size(0) for tensor in src_tensors])
            tgt_max_len = max([tensor.size(0) for tensor in tgt_tensors])

            # Pad to max length in batch
            for i in range(len(src_tensors)):
                src_padding = torch.full(
                    (src_max_len - src_tensors[i].size(0),),
                    self.special_tokens["PAD"],
                    dtype=torch.long,
                )
                tgt_padding = torch.full(
                    (tgt_max_len - tgt_tensors[i].size(0),),
                    self.special_tokens["PAD"],
                    dtype=torch.long,
                )
                src_tensors[i] = torch.cat([src_tensors[i], src_padding])
                tgt_tensors[i] = torch.cat([tgt_tensors[i], tgt_padding])

            # Stack tensors into a batch [batch_size, seq_len]
            src_batch = torch.stack(src_tensors)
            tgt_batch = torch.stack(tgt_tensors)

            # Transpose to [seq_len, batch_size] for RNN processing
            src_batch = src_batch.transpose(0, 1)
            tgt_batch = tgt_batch.transpose(0, 1)

            yield src_batch, tgt_batch


class Trainer:
    """Class for training sequence-to-sequence models"""

    def __init__(
        self,
        encoder,
        decoder,
        dataloader,
        learning_rate=0.005,
        teacher_forcing_ratio=0.5,
        clip_value=0.5,
        device=None,
        accum_steps=1,
    ):
        """
        Initialize trainer

        Args:
            encoder: Encoder model
            decoder: Decoder model
            dataloader: DataLoader instance
            learning_rate: Learning rate for optimizer
            teacher_forcing_ratio: Probability of using teacher forcing
            clip_value: Gradient clipping value
            device: Device to train on (cuda/cpu)
            accum_steps: Steps to accumulate gradients
        """
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.clip_value = clip_value

        # Determine device
        self.device = device
        print(f"Training on: {self.device}")

        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
        )

        # Initialize loss function
        self.criterion = nn.NLLLoss(ignore_index=dataloader.special_tokens["PAD"])

        # Initialize best loss for checkpointing
        self.best_loss = float("inf")
        self.accum_steps = accum_steps

    def process_batch(self, src_batch, tgt_batch, max_length=512):
        """
        Process a single batch

        Args:
            src_batch: Source batch tensor [seq_len, batch_size]
            tgt_batch: Target batch tensor [seq_len, batch_size]
            max_length: Maximum sequence length

        Returns:
            loss: Batch loss
        """
        batch_size = src_batch.size(1)

        # Move tensors to device
        src_batch, tgt_batch = src_batch.to(self.device), tgt_batch.to(self.device)

        # Initialize loss
        loss = 0

        # Zero gradients
        self.optimizer.zero_grad()

        try:
            # Initialize encoder hidden state
            encoder_hidden = self.encoder.get_initial_hidden_state(batch_size)

            # Forward pass through encoder
            encoder_outputs, encoder_hidden = self.encoder(src_batch, encoder_hidden)

            # Prepare decoder input (start with SOS tokens)
            decoder_input = torch.full(
                (batch_size,),
                self.dataloader.special_tokens["SOS"],
                dtype=torch.long,
                device=self.device,
            )

            # Set initial decoder hidden state to final encoder hidden state
            decoder_hidden = encoder_hidden

            # Determine whether to use teacher forcing
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio

            # Maximum number of decoding steps
            target_length = tgt_batch.size(0)
            max_output_length = min(target_length, max_length)

            # Forward pass through decoder
            for t in range(1, max_output_length):  # Start from 1 to skip the SOS token
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, encoder_outputs, decoder_hidden
                )

                # Add to loss
                loss += self.criterion(decoder_output, tgt_batch[t])

                # Determine next input
                if use_teacher_forcing:
                    # Teacher forcing: use target as next input
                    decoder_input = tgt_batch[t]
                else:
                    # Without teacher forcing: use decoder's own prediction
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(
                        -1
                    ).detach()  # Detach from history as input

            # Normalize loss by sequence length
            loss = loss / (max_output_length - 1)  # -1 because we skip the first token

            return loss

        except Exception as e:
            print(f"Error in processing batch: {str(e)}")
            return None

    def update_model(self, loss):
        """
        Update model parameters

        Args:
            loss: Loss tensor

        Returns:
            success: Whether the update was successful
        """
        if loss is None:
            return False

        try:
            # Check for NaN loss
            if torch.isnan(loss):
                print("⚠️ NaN detected in loss!")
                return False

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.clip_value
            )
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), max_norm=self.clip_value
            )

            # Update parameters
            self.optimizer.step()

            return True

        except Exception as e:
            print(f"Error in updating model: {str(e)}")
            return False

    def train_epoch(self, max_length=512):
        """
        Train for one epoch

        Args:
            max_length: Maximum sequence length

        Returns:
            epoch_loss: Average loss for the epoch
        """
        # Set models to training mode
        self.encoder.train()
        self.decoder.train()

        # Initialize counters
        total_loss = 0
        batch_count = 0
        successful_batches = 0

        # Process batches
        step = 0
        self.optimizer.zero_grad()
        for batch_idx, (src_batch, tgt_batch) in enumerate(
            self.dataloader.get_batches(max_length)
        ):
            step += 1
            loss = self.process_batch(src_batch, tgt_batch, max_length)
            if loss is not None:
                (loss / self.accum_steps).backward()
                if (
                    step % self.accum_steps == 0
                    or batch_idx == len(self.dataloader) - 1
                ):
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.decoder.parameters()),
                        self.clip_value,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += loss.item()
                    successful_batches += 1
            batch_count += 1
            if batch_count % 10 == 0:
                if successful_batches > 0:
                    avg_loss = total_loss / successful_batches
                    print(f"  Batch {batch_count} - Avg Loss: {avg_loss:.4f}")
                else:
                    print(f"  Batch {batch_count} - No successful batches yet")
            batch_size_current = src_batch.size(1)  # Get batch size outside try block

            try:
                # Process batch
                loss = self.process_batch(src_batch, tgt_batch, max_length)

                # Update model
                if loss is not None and self.update_model(loss):
                    total_loss += loss.item()
                    successful_batches += 1

                # Increment batch counter
                batch_count += 1

                # Print progress
                if batch_count % 10 == 0:
                    if successful_batches > 0:
                        avg_loss = total_loss / successful_batches
                        print(f"  Batch {batch_count} - Avg Loss: {avg_loss:.4f}")
                    else:
                        print(f"  Batch {batch_count} - No successful batches yet")

            except Exception as e:
                print(f"Error in batch {batch_count}: {str(e)}")
                continue

        # Calculate average loss
        epoch_loss = (
            total_loss / successful_batches if successful_batches > 0 else float("inf")
        )

        return epoch_loss

    def save_checkpoint(self, epoch, loss, checkpoint_path):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch number
            loss: Validation loss
            checkpoint_path: Path to save checkpoint

        Returns:
            saved: Whether the checkpoint was saved
        """
        try:
            # Get model type from encoder
            model_type = "lstm" if hasattr(self.encoder, "lstm") else "gru"

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "src_vocab": self.dataloader.src_vocab,
                "tgt_vocab": self.dataloader.tgt_vocab,
                "src_index2word": self.dataloader.src_index2word,
                "tgt_index2word": self.dataloader.tgt_index2word,
                "special_tokens": self.dataloader.special_tokens,
                "model_type": model_type,
                "hidden_size": self.encoder.hidden_size,
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            return False

    def train(self, num_epochs, checkpoint_path, max_length=512):
        """
        Train the model for multiple epochs

        Args:
            num_epochs: Number of epochs to train
            checkpoint_path: Path to save best checkpoint
            max_length: Maximum sequence length

        Returns:
            best_loss: Best validation loss achieved
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            # Track time
            start_time = time.time()

            try:
                # Train for one epoch
                loss = self.train_epoch(max_length)

                # Calculate epoch time
                epoch_time = time.time() - start_time
                print(
                    f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} - Time: {epoch_time:.2f}s"
                )

                # Check if this is the best model so far
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint(epoch, loss, checkpoint_path)

            except KeyboardInterrupt:
                print("Training interrupted by user.")
                break

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue

        print("Training complete!")
        return self.best_loss


def main():
    """Main function to run training"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a sequence-to-sequence model for machine translation"
    )
    parser.add_argument(
        "--train-file",
        default="data/spm_parallel_corpus.tsv",
        help="Path to training data",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/model_checkpoint.pt",
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        default="lstm",
        choices=["gru", "lstm"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Size of hidden layers"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--teacher-forcing", type=float, default=0.5, help="Teacher forcing ratio"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--accum-steps", type=int, default=1, help="Steps to accumulate gradients"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Initialize data loader
    dataloader = DataLoader(
        data_path=args.train_file,
        batch_size=args.batch_size,
        src_lang_col="Russian",
        tgt_lang_col="English",
        max_vocab_size=30000,  # Set maximum vocabulary size
    )

    # Initialize models using the factory function
    encoder, decoder = create_model(
        model_type=args.model_type,
        input_size=len(dataloader.src_vocab),
        output_size=len(dataloader.tgt_vocab),
        hidden_size=args.hidden_size,
    )

    # Initialize trainer
    trainer = Trainer(
        encoder=encoder,
        accum_steps=args.accum_steps,
        decoder=decoder,
        dataloader=dataloader,
        learning_rate=args.lr,
        teacher_forcing_ratio=args.teacher_forcing,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Check for existing checkpoint
    if os.path.exists(args.checkpoint) and args.resume:
        print(f"Resuming training from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.best_loss = checkpoint.get("loss", float("inf"))

    # Log model type being used
    print(f"Training {args.model_type.upper()} model")

    # Train the model
    trainer.train(args.epochs, args.checkpoint, args.max_length)


if __name__ == "__main__":
    main()
