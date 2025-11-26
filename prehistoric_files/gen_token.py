# @title
import multiprocessing
import madmom
import os
import hashlib
import pickle
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio


def extract_features_worker(args):
    """Worker function for parallel feature extraction"""
    file_path, base_path = args

    try:
        # Generate hash/path
        filename = os.path.basename(file_path)
        file_hash = hashlib.md5(filename.encode()).hexdigest()
        checkpoint_path = f"./checkpoints/{file_hash}.pkl"

        # Skip if checkpoint exists (assuming features are already done)
        if os.path.exists(checkpoint_path):
            return

        # Import locally to ensure worker independence
        import madmom
        from madmom.models import BEATS_LSTM

        # Initialize processors
        key_proc = madmom.features.key.CNNKeyRecognitionProcessor()
        beat_proc = madmom.features.beats.RNNBeatProcessor(
            online=True, nn_files=[BEATS_LSTM[0]]
        )

        # Key recognition
        key_probs = key_proc(file_path)
        key_idx = np.argmax(key_probs)
        # key_idx = 0

        # Beat tracking
        beat_act = beat_proc(file_path, fps=75)

        data = {"key": key_idx, "beat_act": beat_act, "step1_complete": True}

        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f)

    except Exception as e:
        print(f"Error in feature extraction for {file_path}: {e}")


def add_tokens_to_checkpoint(file_path, wavtokenizer, device, sr, channels, base_path):
    """Add tokens to an existing checkpoint"""
    filename = os.path.basename(file_path)
    file_hash = hashlib.md5(filename.encode()).hexdigest()
    checkpoint_path = f"./checkpoints/{file_hash}.pkl"

    if not os.path.exists(checkpoint_path):
        print(
            f"Skipping tokenization for {file_path}: Checkpoint not found (Step 1 failed?)"
        )
        return

    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        if "tokens" in checkpoint:
            return  # Already tokenized

        # Load and convert audio
        wav, sr_orig = torchaudio.load(file_path)
        wav = convert_audio(wav, sr_orig, sr, channels)

        # Process in chunks
        chunk_length = 30 * sr
        all_tokens = []

        for start in range(0, wav.shape[1], chunk_length):
            end = min(start + chunk_length, wav.shape[1])
            chunk = wav[:, start:end].to(device)
            bandwidth_id = torch.tensor([0]).to(device)
            _, discrete_code = wavtokenizer.encode_infer(
                chunk, bandwidth_id=bandwidth_id
            )
            all_tokens.append(discrete_code.cpu())

        tokens = torch.cat(all_tokens, dim=2) if len(all_tokens) > 1 else all_tokens[0]
        tokens = tokens.detach().clone()

        # Update checkpoint
        checkpoint["tokens"] = tokens
        checkpoint["samplerate"] = sr
        checkpoint["channels"] = channels
        checkpoint["source_filepath"] = os.path.relpath(file_path, base_path)

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

    except Exception as e:
        print(f"Error in tokenization for {file_path}: {e}")


def preprocess_dataset():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "./wavtokenizer_music_large.yaml"
    model_path = "./wavtokenizer_music_large.ckpt"
    data_path = "dataset"
    sr = 24000
    channels = 2

    os.makedirs("./checkpoints", exist_ok=True)

    # Gather files
    audio_files = []
    if os.path.exists(data_path):
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".wav", ".mp3")):
                    audio_files.append(os.path.join(root, file))

    if not audio_files:
        print("No audio files found.")
        return

    print(f"Found {len(audio_files)} files to process.")

    # Step 1: Parallel Feature Extraction
    print("\nStep 1/2: Extracting features (Parallel CPU)...")
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores.")

    # Prepare arguments for workers
    worker_args = [(f, data_path) for f in audio_files]

    # Create pool and run
    with multiprocessing.Pool(processes=16) as pool:
        list(
            tqdm(
                pool.imap_unordered(extract_features_worker, worker_args),
                total=len(audio_files),
                desc="Feature Extraction",
                unit="file",
            )
        )

    # Step 2: Sequential Tokenization
    print("\nStep 2/2: Tokenizing audio (Sequential GPU)...")

    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Skipping tokenization.")
        return

    print("Loading WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    for file_path in tqdm(audio_files, desc="Tokenization", unit="file"):
        add_tokens_to_checkpoint(
            file_path, wavtokenizer, device, sr, channels, data_path
        )

    del wavtokenizer
    torch.cuda.empty_cache()
    print("Preprocessing complete.")


if __name__ == "__main__":
    # Run preprocessing
    preprocess_dataset()
