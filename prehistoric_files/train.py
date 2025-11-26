import os
import pickle
import random
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import contextlib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from decoder.pretrained import WavTokenizer
import glob
import bisect
import config as cfg


class TransformerConfig:
    def __init__(self):
        # Existing parameters
        self.base_vocab_size = cfg.Config.base_vocab_size
        self.use_eos_token = cfg.Config.use_eos_token
        self.actual_vocab_size = (
            self.base_vocab_size + 1 if self.use_eos_token else self.base_vocab_size
        )
        self.eos_token_id = self.base_vocab_size if self.use_eos_token else None
        self.d_model = cfg.Config.d_model
        self.n_heads = cfg.Config.n_heads
        self.n_layers = cfg.Config.n_layers
        self.d_ff = cfg.Config.d_ff
        self.context_length = cfg.Config.context_length
        self.dropout = cfg.Config.dropout
        self.lr = cfg.Config.lr
        self.batch_size = cfg.Config.batch_size
        self.epochs = cfg.Config.epochs
        self.save_every = cfg.Config.save_every
        self.eval_every = cfg.Config.eval_every
        self.warmup_steps = cfg.Config.warmup_steps
        self.use_pretrained_embeddings = cfg.Config.use_pretrained_embeddings
        self.freeze_embeddings = cfg.Config.freeze_embeddings

        # New optimization parameters
        self.use_mixed_precision = cfg.Config.use_mixed_precision
        self.gradient_accumulation_steps = cfg.Config.gradient_accumulation_steps
        self.use_8bit_optimizer = cfg.Config.use_8bit_optimizer
        self.freeze_layers = cfg.Config.freeze_layers
        self.use_cudnn_benchmark = cfg.Config.use_cudnn_benchmark


# Improved token embedding that uses codebook weights from tokenizer
class OptimizedTokenEmbedding(nn.Module):
    def __init__(self, config, wavtokenizer=None):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Initialize with pretrained weights if requested
        if config.use_pretrained_embeddings and wavtokenizer is not None:
            print("Initializing embedding weights from WavTokenizer")
            self._initialize_from_tokenizer(wavtokenizer, config)

            # If embeddings should be frozen
            if config.freeze_embeddings:
                print("Freezing embedding weights")
                self.embedding.weight.requires_grad = False
        else:
            print("Using randomly initialized embeddings")
            # Initialize embedding layer with the actual vocab size
            embedding_dim = self.d_model
            self.embedding = nn.Embedding(config.actual_vocab_size, embedding_dim)

    def _initialize_from_tokenizer(self, wavtokenizer, config):
        """Extract codebook weights from wavtokenizer and use them to initialize embedding"""
        print("Extracting embeddings from tokenizer...")

        # Get base embeddings from tokenizer
        embed_weights = wavtokenizer.extract_all_features().to("cpu")

        # Check if we need to extend the embeddings for EOS token
        if config.use_eos_token:
            print(f"Initializing EOS token embedding (token ID: {config.eos_token_id})")
            # Create a new embedding tensor with space for the EOS token
            extended_weights = torch.zeros(config.actual_vocab_size, config.d_model)
            # Copy base embeddings
            extended_weights[: config.base_vocab_size] = embed_weights
            # Use average of all embeddings for EOS, with small noise
            eos_embedding = (
                embed_weights.mean(dim=0) + torch.randn(config.d_model) * 0.02
            )
            extended_weights[config.eos_token_id] = eos_embedding
            embed_weights = extended_weights

        num_embeddings, embedding_dim = embed_weights.shape
        # Check if the embedding dimension matches the model's d_model
        assert (
            embedding_dim == self.d_model
        ), f"Embedding dimension {embedding_dim} does not match d_model {self.d_model}"

        # Create an nn.Embedding layer with the codebook weights.
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(embed_weights)

        print(f"Embedding weights shape: {self.embedding.weight.shape}")

    def forward(self, x):
        # Simple forward pass - just use the embedding layer
        return self.embedding(x)


# Audio token dataset - keep unchanged
class AudioTokenDataset(Dataset):
    def __init__(self, checkpoint_dir, context_length, eos_token_id=None):
        self.checkpoint_dir = checkpoint_dir
        self.context_length = context_length
        self.eos_token_id = eos_token_id
        self.checkpoint_files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".pkl")
        ]
        print(f"Found {len(self.checkpoint_files)} checkpoint files")

        # Pre-load all tokens for faster training
        self.all_tokens = []
        self.sequence_boundaries = [0]  # Store start indices of sequences

        max_token_id = 0  # Track maximum token ID for validation

        for file_path in tqdm(self.checkpoint_files, desc="Loading checkpoints"):
            with open(file_path, "rb") as f:
                checkpoint = pickle.load(f)
                tokens = checkpoint["tokens"]  # shape [1, 2, seq_len]

                # Track max token ID
                max_token_id = max(max_token_id, tokens.max().item())

                # Reorganize tokens to interleave left and right channels
                # from [1, 2, seq_len] to [1, seq_len*2]
                left_channel = tokens[0, 0, :]  # First channel
                right_channel = tokens[0, 1, :]  # Second channel

                # Interleave channels
                interleaved = torch.empty(left_channel.size(0) * 2, dtype=torch.long)
                interleaved[0::2] = left_channel
                interleaved[1::2] = right_channel

                # If using EOS tokens, append to each sequence
                if self.eos_token_id is not None:
                    eos_tokens = torch.tensor(
                        [self.eos_token_id, self.eos_token_id], dtype=torch.long
                    )
                    interleaved = torch.cat([interleaved, eos_tokens])

                self.all_tokens.append(interleaved)
                self.sequence_boundaries.append(
                    self.sequence_boundaries[-1] + len(interleaved)
                )

        # Concatenate all tokens
        self.tokens = torch.cat(self.all_tokens)

        print(f"Total token sequence length: {len(self.tokens)}")
        print(f"Number of audio sequences: {len(self.sequence_boundaries) - 1}")
        print(f"Maximum token ID in dataset: {max_token_id}")

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.context_length)

    def __getitem__(self, idx):
        # Calculate start index with stride = context_length
        start_idx = idx * self.context_length

        # Get a sequence of context_length tokens
        x = self.tokens[start_idx : start_idx + self.context_length]
        # Target is the next token in the sequence
        y = self.tokens[start_idx + 1 : start_idx + self.context_length + 1]

        # If using EOS tokens, we need to check if we're crossing sequence boundaries
        if self.eos_token_id is not None:
            # Find which sequence start_idx belongs to using binary search
            seq_idx = bisect.bisect_right(self.sequence_boundaries, start_idx) - 1

            # Check if we're crossing a sequence boundary
            next_boundary = self.sequence_boundaries[seq_idx + 1]
            if start_idx + self.context_length >= next_boundary:
                # We're crossing a boundary, pad with EOS tokens
                x_end_offset = next_boundary - start_idx
                pad_len = self.context_length - x_end_offset

                # Ensure padding length is non-negative
                if pad_len > 0:
                    eos_padding = torch.full(
                        (pad_len,), self.eos_token_id, dtype=torch.long
                    )
                    x = torch.cat([x[:x_end_offset], eos_padding])
                    y = torch.cat([y[:x_end_offset], eos_padding])

        # Make sure tensors are exactly context_length
        if len(x) < self.context_length:
            padding = torch.full(
                (self.context_length - len(x),),
                self.eos_token_id if self.eos_token_id is not None else 0,
                dtype=torch.long,
            )
            x = torch.cat([x, padding])

        if len(y) < self.context_length:
            padding = torch.full(
                (self.context_length - len(y),),
                self.eos_token_id if self.eos_token_id is not None else 0,
                dtype=torch.long,
            )
            y = torch.cat([y, padding])

        return x, y


class EfficientAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert (
            self.head_dim * n_heads == d_model
        ), "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # scaled_dot_product_attention expects [batch, heads, seq_len, head_dim]
        # It handles causal masking if is_causal=True
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(output)


class EfficientTransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EfficientAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # Attention block
        attn_output = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward block
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Simplified TransformerModel that uses the optimized embedding
class TransformerModel(nn.Module):
    def __init__(self, config, wavtokenizer=None):
        super().__init__()
        self.config = config

        # Token embedding - now using the optimized embedding layer
        self.token_embedding = OptimizedTokenEmbedding(config, wavtokenizer)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.dropout, config.context_length
        )

        # Create a stack of sparse transformer layers
        self.transformer_layers = nn.ModuleList(
            [EfficientTransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )

        # Use actual_vocab_size for the output layer to include EOS token if needed
        self.output_layer = nn.Linear(config.d_model, config.actual_vocab_size)

        # Initialize non-embedding weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # Only initialize output layer since embeddings are handled separately
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x shape: [batch_size, context_length]

        # Create embedding
        x = self.token_embedding(x)  # [batch_size, context_length, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # The mask is now handled inside the CustomAttention layer
        for layer in self.transformer_layers:
            x = layer(x)

        output = self.output_layer(x)

        return output


# Rest of the code - PositionalEncoding, get_lr_scheduler - remains unchanged
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (not a model parameter, but part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def get_lr_scheduler(optimizer, warmup_steps, d_model):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1.0, warmup_steps)
        return 1.0  # Constant learning rate after warmup

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def freeze_model_layers(model, num_layers_to_freeze):
    """Freeze the bottom transformer layers of the model"""
    print(f"Freezing bottom {num_layers_to_freeze} transformer layers")

    # Freeze embedding if configured
    if model.config.freeze_embeddings:
        for param in model.token_embedding.parameters():
            param.requires_grad = False

    # Freeze specific transformer layers
    if num_layers_to_freeze > 0:
        for i in range(min(num_layers_to_freeze, model.config.n_layers)):
            for param in model.transformer_encoder.layers[i].parameters():
                param.requires_grad = False
            print(f"  Transformer layer {i} frozen")


# Modified to save and load tokenizer embeddings
def train(config):
    # Set device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Enable cudnn benchmark for better performance if requested
    if config.use_cudnn_benchmark and device.type == "cuda":
        print("Enabling cuDNN benchmark for potentially faster training")
        torch.backends.cudnn.benchmark = True

    # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling TF32 for faster training")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # Create output directories
    os.makedirs("./model_checkpoints", exist_ok=True)
    os.makedirs("./generated_samples", exist_ok=True)
    os.makedirs(
        "./model_checkpoints/frequent", exist_ok=True
    )  # For frequent checkpoints

    # Load wavtokenizer only for initialization, then save its embeddings
    config_path = "./wavtokenizer_music_large.yaml"
    model_path = "./wavtokenizer_music_large.ckpt"
    print("Loading WavTokenizer for embedding initialization...")
    tokenizer_for_init = WavTokenizer.from_pretrained0802(config_path, model_path)
    tokenizer_for_init = tokenizer_for_init.to(device)

    # Create model with the tokenizer (will initialize embeddings)
    model = TransformerModel(config, tokenizer_for_init).to(device)

    # Freeze layers if requested
    if config.freeze_layers > 0:
        freeze_model_layers(model, config.freeze_layers)

    # Compile model for faster training (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # After model creation, release the tokenizer to free memory
    del tokenizer_for_init
    torch.cuda.empty_cache()

    # For generation and evaluation, we'll still need the tokenizer
    # but we'll load it only when needed
    tokenizer_file_paths = {"config_path": config_path, "model_path": model_path}

    # Setup optimizer based on configuration
    if config.use_8bit_optimizer and device.type == "cuda":
        try:
            import bitsandbytes as bnb  # type: ignore

            print("Using 8-bit AdamW optimizer")
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
            )
        except ImportError:
            print(
                "Warning: bitsandbytes not installed. Falling back to standard AdamW."
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
        )

    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.d_model)

    # Setup mixed precision training
    scaler = (
        torch.cuda.amp.GradScaler()
        if config.use_mixed_precision and device.type in ["cuda", "xpu"]
        else None
    )
    if scaler:
        print("Using mixed precision training")

    # Check for an existing frequent checkpoint and resume if present
    frequent_ckpt_files = glob.glob("./model_checkpoints/frequent/*.pt")
    if frequent_ckpt_files:
        latest_ckpt = max(frequent_ckpt_files, key=os.path.getmtime)
        print(f"Resuming from frequent checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get(
            "global_step", 0
        )  # Resume global step for sample generation
    else:
        start_epoch = 0
        global_step = 0

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    batch_count = 0  # New counter for total batches processed
    best_loss = float("inf")

    # Track accumulated gradients and loss
    accumulated_loss = 0

    # List to keep track of frequent checkpoint files
    frequent_checkpoints = []

    # Create dataset and dataloader
    eos_token_id = config.eos_token_id if config.use_eos_token else None
    dataset = AudioTokenDataset("./checkpoints", config.context_length, eos_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # Pre-load the tokenizer for generation to avoid reloading in the loop
    print("Loading WavTokenizer for sample generation...")
    wavtokenizer_for_gen = WavTokenizer.from_pretrained0802(
        tokenizer_file_paths["config_path"],
        tokenizer_file_paths["model_path"],
    ).to(device)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        save_interval = config.save_every
        print(
            f"Will save frequent checkpoint every {save_interval} batches (Epoch {epoch+1})"
        )

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for step, (x, y) in enumerate(progress_bar):
            batch_count += 1
            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision if enabled
            autocast_kwargs = {}
            if device.type == "xpu":
                autocast_kwargs["dtype"] = torch.bfloat16
            with (
                torch.amp.autocast(device_type=device.type, **autocast_kwargs)
                if config.use_mixed_precision and device.type in ["cuda", "xpu"]
                else contextlib.nullcontext()
            ):
                logits = model(x)
                logits = logits.view(-1, config.actual_vocab_size)
                y = y.view(-1)
                loss = criterion(logits, y)
                loss = loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0 or step == len(
                dataloader
            ) - 1:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += accumulated_loss
                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {"loss": f"{accumulated_loss:.4f}", "lr": f"{current_lr:.6f}"}
                )
                global_step += 1
                accumulated_loss = 0

            if step > 0 and step % save_interval == 0:
                frequent_save_path = (
                    f"./model_checkpoints/frequent/model_e{epoch+1}_step{step}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": vars(config),
                        "global_step": global_step,
                    },
                    frequent_save_path,
                )
                frequent_checkpoints.append(frequent_save_path)
                if len(frequent_checkpoints) > 3:
                    oldest_checkpoint = frequent_checkpoints.pop(0)
                    if os.path.exists(oldest_checkpoint):
                        os.remove(oldest_checkpoint)
                        print(f"Removed old checkpoint: {oldest_checkpoint}")
                print(f"Frequent checkpoint saved to {frequent_save_path}")

            if global_step > 0 and global_step % config.eval_every == 0:
                sample_tokens = generate_sample(
                    model,
                    wavtokenizer_for_gen,
                    config,
                    device,
                    temperature=0.7,
                    stop_on_eos=True,
                )
                save_generated_audio(
                    sample_tokens,
                    wavtokenizer_for_gen,
                    device,
                    f"./generated_samples/sample_step_{global_step}.wav",
                )
                model.train()

        epoch_loss /= len(dataloader) / config.gradient_accumulation_steps
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} completed in {elapsed:.2f}s - Avg Loss: {epoch_loss:.4f}"
        )

        if (epoch + 1) % 1 == 0:
            save_path = f"./model_checkpoints/model_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": epoch_loss,
                    "config": vars(config),
                },
                save_path,
            )
            print(f"Model saved to {save_path}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = f"./model_checkpoints/model_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": epoch_loss,
                        "config": vars(config),
                    },
                    best_path,
                )
                print(f"New best model saved to {best_path}")

    print("Training completed!")


# Generation functions remain largely the same, but load tokenizer when needed
def generate_sample(
    model,
    wavtokenizer,
    config,
    device,
    start_tokens=None,
    sample_length=2250,  # ~30 sec at 70 token/s
    temperature=1.0,
    stop_on_eos=True,
    top_k=0,  # Top-k sampling
    repetition_penalty=1.3,  # Add repetition penalty (1.0 = no penalty)
    repetition_window=100,  # Consider only recent tokens for penalty
    debug=False,
):
    model.eval()

    # If no start tokens provided, create random ones (improved initialization)
    if start_tokens is None:
        # Start with 16 random tokens (8 for each channel) for better conditioning
        start_tokens = torch.randint(0, config.base_vocab_size, (1, 16), device=device)

    # Ensure start_tokens is on the correct device
    start_tokens = start_tokens.to(device)

    # Generate tokens autoregressively
    generated = start_tokens
    eos_generated = False

    with torch.no_grad():
        for i in tqdm(range(sample_length), desc="Generating"):
            # Get sequence context (up to context_length)
            context = generated[:, -config.context_length :]

            # Get predictions
            logits = model(context)
            logits = logits[:, -1, :]  # Get the logits for next token prediction

            # Apply temperature
            logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                # Get the recent tokens to penalize (within repetition window)
                recent_tokens = generated[
                    :, max(0, generated.size(1) - repetition_window) :
                ]

                # Create a set of unique tokens to penalize
                penalty_tokens = torch.unique(recent_tokens)

                # Apply penalty to these tokens
                if penalty_tokens.size(0) > 0:
                    for token_id in penalty_tokens:
                        # Penalize by dividing or multiplying logit by penalty factor
                        # depending on whether the logit is positive or negative
                        token_logit = logits[0, token_id]
                        if token_logit > 0:
                            logits[0, token_id] = token_logit / repetition_penalty
                        else:
                            logits[0, token_id] = token_logit * repetition_penalty

                    if debug and i % 100 == 0:
                        print(
                            f"Applied repetition penalty to {penalty_tokens.size(0)} tokens"
                        )

            # Optional top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from the distribution for the last token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Debug info
            if debug and i % 100 == 0:
                print(f"Token {i}: Generated token {next_token.item()}")

            # Add to generated
            generated = torch.cat([generated, next_token], dim=1)

            # If using EOS tokens and we've generated one, track it
            if config.use_eos_token and next_token.item() == config.eos_token_id:
                eos_generated = True
                if debug:
                    print(f"EOS token generated at position {i}")
                if stop_on_eos:
                    print("EOS token generated, stopping generation")
                    break

    # Rest of the function remains unchanged
    if debug:
        print(f"Final sequence length: {generated.size(1)} tokens")
        if config.use_eos_token:
            print(f"EOS tokens generated: {eos_generated}")

    # Separate left and right channels
    num_tokens = generated.size(1)
    left_channel = generated[0, 0::2]  # Even indices
    right_channel = generated[0, 1::2]  # Odd indices

    # Make channels the same length (take the min length to ensure both are valid)
    min_len = min(left_channel.size(0), right_channel.size(0))
    left_channel = left_channel[:min_len]
    right_channel = right_channel[:min_len]

    if debug:
        print(f"Channel lengths: {left_channel.size(0)} L, {right_channel.size(0)} R")

    # Handle EOS tokens if needed
    if config.use_eos_token:
        # Find first occurrence of EOS in each channel
        left_eos = (left_channel == config.eos_token_id).nonzero(as_tuple=True)[0]
        right_eos = (right_channel == config.eos_token_id).nonzero(as_tuple=True)[0]

        # Only trim if EOS tokens are found in both channels
        if left_eos.numel() > 0 and debug:
            print(f"Left channel EOS at position {left_eos[0].item()}")

        if right_eos.numel() > 0 and debug:
            print(f"Right channel EOS at position {right_eos[0].item()}")

        # Trim at EOS only if stop_on_eos is True and the EOS is found
        if stop_on_eos:
            if left_eos.numel() > 0:
                left_channel = left_channel[: left_eos[0]]
            if right_eos.numel() > 0:
                right_channel = right_channel[: right_eos[0]]

        # Make channels the same length again after trimming
        min_len = min(left_channel.size(0), right_channel.size(0))
        left_channel = left_channel[:min_len]
        right_channel = right_channel[:min_len]

    # Reshape to [1, 2, seq_len] for wavtokenizer
    tokens_for_decode = torch.stack([left_channel, right_channel], dim=0).unsqueeze(0)

    if debug:
        print(f"Final audio length: {min_len/70:.2f} seconds at 70 tokens/s")

    return tokens_for_decode


# Save generated audio
def save_generated_audio(tokens, wavtokenizer, device, output_path, sr=24000):
    # Convert tokens to features for decoder
    tokens = tokens.to(device)
    features = wavtokenizer.codes_to_features(tokens).to(device)
    bandwidth_id = torch.tensor([0]).to(device)

    # Decode audio
    audio = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

    # Save as WAV
    torchaudio.save(
        output_path,
        audio.cpu(),
        sample_rate=sr,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    print(f"Generated audio saved to {output_path}")


def main():
    # Create config from args
    config = TransformerConfig()

    # Print vocabulary info
    print(f"Base vocabulary size: {config.base_vocab_size}")
    print(f"Using EOS tokens: {config.use_eos_token}")
    if config.use_eos_token:
        print(f"EOS token ID: {config.eos_token_id}")
    print(f"Actual vocabulary size: {config.actual_vocab_size}")

    # Print optimization settings
    print(f"Mixed precision training: {config.use_mixed_precision}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Using 8-bit optimizer: {config.use_8bit_optimizer}")
    print(f"Freezing {config.freeze_layers} transformer layers")
    print(f"Using cuDNN benchmark: {config.use_cudnn_benchmark}")

    # Start training
    train(config)


if __name__ == "__main__":
    main()
