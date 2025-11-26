import os
import random
import torch
import torchaudio
import pickle
import tempfile
import shutil
from tqdm import tqdm
from gen_token import process_audio_file
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer


def setup_environment():
    """Set up test environment and return configuration and resources."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "./wavtokenizer_music_large.yaml"
    model_path = "./wavtokenizer_music_large.ckpt"
    sr = 24000
    channels = 2

    # Create test directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./test_audio", exist_ok=True)

    # Load model
    print("Loading model...")
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    # Create a small test audio file
    test_audio_path = "./test_audio/test_sample.wav"
    if not os.path.exists(test_audio_path):
        # Generate 3 seconds of white noise as test audio
        sample_rate = 24000
        dummy_waveform = torch.randn(2, sample_rate * 3)  # 2 channels, 3 seconds
        torchaudio.save(test_audio_path, dummy_waveform, sample_rate)

    base_path = "./test_audio"
    return {
        "device": device,
        "config_path": config_path,
        "model_path": model_path,
        "sr": sr,
        "channels": channels,
        "wavtokenizer": wavtokenizer,
        "test_audio_path": test_audio_path,
        "base_path": base_path,
    }


def test_checkpoint_to_wav(config):
    """Test loading a random checkpoint and converting it back to WAV."""
    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir("./checkpoints") if f.endswith(".pkl")]

    if not checkpoint_files:
        print("No checkpoint files found. Please run test_process_audio_file first.")
        return

    # Select a random checkpoint
    random_checkpoint = random.choice(checkpoint_files)
    checkpoint_path = os.path.join("./checkpoints", random_checkpoint)

    print(f"Converting checkpoint {random_checkpoint} to audio")

    # Load the checkpoint
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract tokens and other information
    tokens = checkpoint["tokens"]
    sr = checkpoint["samplerate"]

    # Process tokens in chunks
    chunk_length = 2250  # 30 seconds worth of tokens (75 tokens/sec * 30 sec)
    processed_chunks = []

    for start in range(0, tokens.shape[2], chunk_length):
        end = min(start + chunk_length, tokens.shape[2])
        token_chunk = tokens[:, :, start:end].to(config["device"])
        print(f"last token: {token_chunk.shape}")
        features = (
            config["wavtokenizer"].codes_to_features(token_chunk).to(config["device"])
        )
        bandwidth_id = torch.tensor([0]).to(config["device"])

        # Convert tokens back to audio
        audio_chunk = config["wavtokenizer"].decode(features, bandwidth_id=bandwidth_id)
        processed_chunks.append(audio_chunk.cpu())

    # Concatenate all processed chunks
    audio_out = (
        torch.cat(processed_chunks, dim=1)
        if len(processed_chunks) > 1
        else processed_chunks[0]
    )

    # Save as WAV file
    output_path = "./test.wav"
    torchaudio.save(
        output_path,
        audio_out.cpu(),
        sample_rate=sr,
        encoding="PCM_S",
        bits_per_sample=16,
    )

    if os.path.exists(output_path):
        print(f"Converted audio saved to {output_path}")
    else:
        print("Failed to save converted audio.")


def main():
    config = setup_environment()


def main():
    config = setup_environment()
    test_checkpoint_to_wav(config)


if __name__ == "__main__":
    main()
