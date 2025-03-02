import os
import re
import json
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
from functools import partial
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import argparse
import torch.multiprocessing as mp

from gpt_download import download_and_load_gpt2
from utils import GPTModel, load_weights_into_gpt

torch.manual_seed(123)


# ------------------------------------------------------------------------------
# Text Generation Functions
# ------------------------------------------------------------------------------


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generate text tokens using a trained model.

    Args:
        model: The language model
        idx: Initial token indices (B, T)
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context size supported by the model

    Returns:
        The generated token sequence
    """
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on last time step
        logits = logits[:, -1, :]

        # Get the highest probability token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs.

    Args:
        text: Input text string
        tokenizer: Tokenizer to use

    Returns:
        Tensor of token IDs with batch dimension
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to text.

    Args:
        token_ids: Tensor of token IDs
        tokenizer: Tokenizer to use

    Returns:
        Decoded text string
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a sample text from the model.

    Args:
        model: The language model
        tokenizer: Tokenizer to use
        device: Device to run inference on
        start_context: Initial text prompt
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


# ------------------------------------------------------------------------------
# Loss Calculation Functions
# ------------------------------------------------------------------------------


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch.

    Args:
        input_batch: Input tokens
        target_batch: Target tokens
        model: The language model
        device: Device to run calculation on

    Returns:
        Calculated loss
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss across multiple batches.

    Args:
        data_loader: DataLoader with batches
        model: The language model
        device: Device to run calculation on
        num_batches: Number of batches to evaluate

    Returns:
        Average loss
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = (
        min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    )

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# ------------------------------------------------------------------------------
# Training and Evaluation Functions
# ------------------------------------------------------------------------------


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate the model on training and validation data.

    Args:
        model: The language model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run evaluation on
        eval_iter: Number of batches to evaluate

    Returns:
        Training and validation loss
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ------------------------------------------------------------------------------
# Data Preparation Functions
# ------------------------------------------------------------------------------


def format_entry(entry):
    """Format a dataset entry into instruction-based format.

    Args:
        entry: Dataset entry with instruction, input, and output fields

    Returns:
        Formatted text string
    """
    instruction_text = (
        f"Below is an instruction that descibes a task."
        f" Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    """Dataset for instruction-based fine-tuning.

    Attributes:
        data: Raw dataset entries
        encoded_texts: Tokenized texts
    """

    def __init__(self, data, tokenizer):
        """Initialize the dataset.

        Args:
            data: List of data entries
            tokenizer: Tokenizer to use
        """
        self.data = data

        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_entry(entry)
            respose_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + respose_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def collate_fn_3(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    """Collate function for DataLoader.

    Args:
        batch: Batch of tokenized texts
        pad_token_id: Token ID for padding
        ignore_index: Index to ignore in loss calculation
        allowed_max_length: Maximum sequence length
        device: Device to place tensors on

    Returns:
        Batched input and target tensors
    """
    # Find longest sequence
    batch_max_length = max(len(item) + 1 for item in batch)

    # Prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add end token
        new_item += [pad_token_id]
        # Pad to max length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Inputs (all but last token)
        targets = torch.tensor(padded[1:])  # Targets (shifted by 1)

        # Replace padding tokens in targets with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Truncate if needed
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Stack and transfer to device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# ------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Plot training and validation losses.

    Args:
        epochs_seen: Epoch numbers
        tokens_seen: Number of tokens processed
        train_losses: Training loss values
        val_losses: Validation loss values
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot losses against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Integer x-axis labels

    # Second x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for alignment
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("visualization/loss-plot.pdf")
    plt.show()


# ------------------------------------------------------------------------------
# Distributed Training Setup
# ------------------------------------------------------------------------------


def ddp_setup(rank, world_size):
    """Setup for distributed data parallel training.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def prepare_dataloader(dataset, batch_size, is_ddp=False, rank=0, world_size=1):
    """Prepare dataloader for training.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        is_ddp: Whether using distributed training
        rank: Process rank
        world_size: Total number of processes

    Returns:
        Configured DataLoader
    """
    if is_ddp:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            sampler=sampler,
            collate_fn=partial(
                collate_fn_3, device="cuda:" + str(rank), allowed_max_length=1024
            ),
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            collate_fn=partial(
                collate_fn_3,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                allowed_max_length=1024,
            ),
        )


# ------------------------------------------------------------------------------
# Trainer Class
# ------------------------------------------------------------------------------


class LMTrainer:
    """Trainer class for language models with distributed training support.

    Attributes:
        model: Language model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for model training
        device: Device to run training on
        is_ddp: Whether using distributed training
        epoch_freq: Frequency to evaluate and save model
        tokenizer: Tokenizer to use
        rank: Process rank
        snapshot_path: Path to save/load checkpoints
        epochs_run: Number of epochs already run
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epoch_freq=1,
        tokenizer=None,
        rank=0,
        is_ddp=False,
        snapshot_path=None,
    ):
        """Initialize trainer.

        Args:
            model: Language model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for model parameters
            device: Device to run training on
            epoch_freq: Frequency to evaluate and save model
            tokenizer: Tokenizer for generating samples
            rank: Process rank
            is_ddp: Whether using distributed training
            snapshot_path: Path to save/load checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.epoch_freq = epoch_freq
        self.tokenizer = tokenizer
        self.rank = rank
        self.is_ddp = is_ddp
        self.snapshot_path = snapshot_path
        self.epochs_run = 0

        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[device])

        if self.snapshot_path and os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        """Load training checkpoint.

        Args:
            snapshot_path: Path to the checkpoint file
        """
        loc = f"cuda:{self.device}"
        snapshot = torch.load(snapshot_path, map_location=loc)

        # Handle DDP model state dict
        if self.is_ddp:
            # Remove 'module.' prefix from keys if present
            state_dict = {
                k.replace("module.", ""): v for k, v in snapshot["MODEL_STATE"].items()
            }
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])

        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(
            f"[GPU{self.rank}] Resuming training from snapshot at Epoch {self.epochs_run}"
        )

    def _save_snapshot(self, epoch, train_losses, val_losses, tokens_seen):
        """Save training checkpoint.

        Args:
            epoch: Current epoch number
            train_losses: Training loss history
            val_losses: Validation loss history
            tokens_seen: Tokens seen during training
        """
        # Only save from rank 0
        if self.rank != 0:
            return

        # Handle DDP model state dict
        if self.is_ddp:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        snapshot = {
            "MODEL_STATE": model_state,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch + 1,
            "TRAIN_LOSSES": train_losses,
            "VAL_LOSSES": val_losses,
            "TOKENS_SEEN": tokens_seen,
        }

        save_path = self.snapshot_path if self.snapshot_path else "checkpoint.pt"
        torch.save(snapshot, save_path)
        print(
            f"[GPU{self.rank}] Epoch {epoch} | Training snapshot saved at {save_path}"
        )

    def train(self, num_epochs):
        """Train the model.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training losses, validation losses, and tokens processed
        """
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen = 0
        best_val_loss = float("inf")

        for epoch in range(self.epochs_run, num_epochs):
            self.model.train()
            epoch_start_time = time.time()

            # Set epoch for distributed sampler
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            for batch_idx, (input_batch, target_batch) in enumerate(self.train_loader):
                input_batch, target_batch = input_batch.to(
                    self.device
                ), target_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), target_batch.flatten()
                )
                loss.backward()
                self.optimizer.step()

                tokens_seen += input_batch.numel()

                # Print progress every 100 batches
                if self.rank == 0 and batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}"
                    )

            # Evaluate after each epoch
            if epoch % self.epoch_freq == 0:
                train_loss, val_loss = evaluate_model(
                    self.model if not self.is_ddp else self.model.module,
                    self.train_loader,
                    self.val_loader,
                    self.device,
                    min(5, len(self.val_loader)),
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # Only print from rank 0
                if self.rank == 0:
                    epoch_time = time.time() - epoch_start_time
                    print(
                        f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                        f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
                    )

                    # Generate sample from rank 0
                    if self.tokenizer and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        sample_model = (
                            self.model if not self.is_ddp else self.model.module
                        )
                        start_context = format_entry(
                            self.val_data[0]
                            if hasattr(self, "val_data")
                            else {"instruction": "Tell me a short story", "input": ""}
                        )
                        generate_and_print_sample(
                            sample_model, self.tokenizer, self.device, start_context
                        )

                # Save snapshot
                self._save_snapshot(epoch, train_losses, val_losses, track_tokens_seen)

        return train_losses, val_losses, track_tokens_seen


# ------------------------------------------------------------------------------
# Main Training Functions
# ------------------------------------------------------------------------------


def load_dataset_and_tokenizer():
    """Load and prepare dataset and tokenizer.

    Returns:
        Data splits, tokenizer
    """
    load_dotenv()

    DATA = os.environ.get("DATA")

    # Load data
    with open(DATA, "r") as f:
        data = json.load(f)

    # Split dataset
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = (
        len(data) - train_portion - test_portion
    )  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    # Set up tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    return train_data, val_data, test_data, tokenizer


def train_single_gpu(args):
    """Train model on a single GPU.

    Args:
        args: Command-line arguments
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load data
    train_data, val_data, test_data, tokenizer = load_dataset_and_tokenizer()

    # Create datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    # Create dataloaders
    batch_size = args.batch_size

    train_loader = prepare_dataloader(train_dataset, batch_size, is_ddp=False)
    val_loader = prepare_dataloader(val_dataset, batch_size, is_ddp=False)

    # Load model
    config = get_model_config(args.model_size)
    model_size = config.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(get_base_config(config))
    load_weights_into_gpt(model, params)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )

    # Create trainer
    trainer = LMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epoch_freq=args.save_every,
        tokenizer=tokenizer,
        snapshot_path=args.snapshot_path,
    )

    # Train
    start_time = time.time()
    train_losses, val_losses, tokens_seen = trainer.train(args.total_epochs)
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")

    # Plot results
    epochs_tensor = torch.linspace(0, args.total_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Save final model
    file_name = f"{re.sub(r'[ ()]', '', args.model_size)}-sft.pth"
    save_dir = os.environ.get("WEIGHT_SAVE_DIR", "./")
    torch.save(model.state_dict(), save_dir + file_name)
    print(f"Model saved as {file_name}")


def train_distributed(rank, world_size, args):
    """Train model in distributed mode.

    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command-line arguments
    """
    # Setup DDP
    if args.mode == "multigpu":
        ddp_setup(rank, world_size)
    elif args.mode == "multigpu_torchrun" or args.mode == "multinode":
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "LOCAL_RANK not found in environment variables. Use torchrun to launch this script."
            )
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl")

    device = rank
    print(f"[GPU{rank}] Process initialized.")

    # Load data
    train_data, val_data, test_data, tokenizer = load_dataset_and_tokenizer()

    # Create datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    # Create dataloaders
    batch_size = args.batch_size

    train_loader = prepare_dataloader(
        train_dataset, batch_size, is_ddp=True, rank=rank, world_size=world_size
    )
    val_loader = prepare_dataloader(
        val_dataset, batch_size, is_ddp=True, rank=rank, world_size=world_size
    )

    # Load model
    config = get_model_config(args.model_size)
    model_size = config.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(get_base_config(config))
    load_weights_into_gpt(model, params)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )

    # Create trainer
    trainer = LMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epoch_freq=args.save_every,
        tokenizer=tokenizer if rank == 0 else None,
        rank=rank,
        is_ddp=True,
        snapshot_path=args.snapshot_path,
    )
    trainer.val_data = val_data  # Attach val_data for sample generation

    # Train
    start_time = time.time()
    train_losses, val_losses, tokens_seen = trainer.train(args.total_epochs)
    end_time = time.time()

    if rank == 0:
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")

        # Plot results
        epochs_tensor = torch.linspace(0, args.total_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

        # Save final model
        file_name = f"{re.sub(r'[ ()]', '', args.model_size)}-sft.pth"
        save_dir = os.environ.get("WEIGHT_SAVE_DIR", "./")
        model_to_save = trainer.model.module
        torch.save(model_to_save.state_dict(), save_dir + file_name)
        print(f"Model saved as {file_name}")

    # Cleanup
    if args.mode != "single_gpu":
        destroy_process_group()


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def get_model_config(model_name):
    """Get model configuration based on name.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration string
    """
    model_configs = {
        "gpt2-small": "gpt2-small (124M)",
        "gpt2-medium": "gpt2-medium (355M)",
        "gpt2-large": "gpt2-large (774M)",
        "gpt2-xl": "gpt2-xl (1558M)",
    }
    return model_configs.get(model_name, "gpt2-medium (355M)")


def get_base_config(model_config):
    """Get base configuration for model.

    Args:
        model_config: Model configuration string

    Returns:
        Configuration dictionary
    """
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[model_config])
    return BASE_CONFIG


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Language model distributed training")
    parser.add_argument(
        "--mode",
        type=str,
        default="single_gpu",
        choices=["single_gpu", "multigpu", "multigpu_torchrun", "multinode"],
        help="Training mode",
    )
    parser.add_argument(
        "--total_epochs", type=int, default=2, help="Total epochs to train the model"
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="How often to save a snapshot"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Input batch size on each device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-medium",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Size of GPT-2 model to use",
    )
    parser.add_argument(
        "--snapshot_path", type=str, default=None, help="Path to save/load checkpoints"
    )
    args = parser.parse_args()

    if args.mode == "single_gpu":
        train_single_gpu(args)
    elif args.mode == "multigpu":
        world_size = torch.cuda.device_count()
        mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size)
    elif args.mode == "multigpu_torchrun" or args.mode == "multinode":
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "LOCAL_RANK not found in environment variables. Use torchrun to launch this script."
            )
        train_distributed(None, None, args)


if __name__ == "__main__":
    main()
