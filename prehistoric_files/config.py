class Config:
    # Model architecture
    base_vocab_size = 4096
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    context_length = 1024
    dropout = 0.1

    # Training hyperparameters
    lr = 1e-3
    batch_size = 8
    epochs = 10
    warmup_steps = 500

    # Checkpointing and Evaluation
    save_every = 1000
    eval_every = 1000

    # Tokenizer settings
    use_eos_token = True
    use_pretrained_embeddings = True
    freeze_embeddings = False

    # Optimization
    use_mixed_precision = True
    gradient_accumulation_steps = 4
    use_8bit_optimizer = False
    freeze_layers = 0
    use_cudnn_benchmark = True
