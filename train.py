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

from gpt_download import download_and_load_gpt2
from utils import GPTModel, load_weights_into_gpt

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


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """Train the language model.

    Args:
        model: The language model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for updating weights
        device: Device to run training on
        num_epochs: Number of epochs to train
        eval_freq: Frequency of evaluation (steps)
        eval_iter: Number of batches for evaluation
        start_context: Text context for generation samples
        tokenizer: Tokenizer to use

    Returns:
        Training losses, validation losses, and tokens processed
    """
    # Initialize tracking lists
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Generate sample after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


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
# Main Execution
# ------------------------------------------------------------------------------

# Load environment variables
load_dotenv()

DATA = os.environ.get("DATA")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
WEIGHT_SAVE_DIR = os.environ.get("WEIGHT_SAVE_DIR")

# Set random seed
torch.manual_seed(123)

# Load data
with open(DATA, "r") as f:
    data = json.load(f)

# Split dataset
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)  # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# Set up tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Create custom collate function
customized_collate_fn = partial(collate_fn_3, device=device, allowed_max_length=1024)

# Create datasets and dataloaders
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn_3,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn_3,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn_3,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
)

# Model configuration
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

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Load model weights
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# Initialize and set up model
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.to(device)

# Training setup
start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

# Train the model
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=format_entry(val_data[0]),
    tokenizer=tokenizer,
)

# Training summary
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Plot results
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Save model
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), WEIGHT_SAVE_DIR + file_name)
print(f"Model saved as {file_name}")
