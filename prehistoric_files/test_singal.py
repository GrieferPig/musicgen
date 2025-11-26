from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


device = torch.device("cuda")

config_path = "./wavtokenizer_music_large.yaml"
model_path = "./wavtokenizer_music_large.ckpt"
audio_source = "./experiments/24121.mp3"
sr = 24000
channels = 2

audio_outpath = f"./experiments/{audio_source.split('/')[-1].split('.')[0]}_vqvae.wav"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


wav, sr_orig = torchaudio.load(audio_source)
wav = convert_audio(wav, sr_orig, sr, channels)

# Calculate one minute in samples (60 seconds)
chunk_length = 30 * sr  # 60 seconds * sample rate

processed_chunks = []
for start in range(0, wav.shape[1], chunk_length):
    print(
        f"Processing chunk {start // chunk_length + 1} of {wav.shape[1] // chunk_length}"
    )
    end = start + chunk_length
    chunk = wav[:, start:end]
    chunk = chunk.to(device)
    bandwidth_id = torch.tensor([0]).to(device)

    features, discrete_code = wavtokenizer.encode_infer(
        chunk, bandwidth_id=bandwidth_id
    )
    audio_chunk = wavtokenizer.decode(features, bandwidth_id=bandwidth_id).cpu()
    processed_chunks.append(audio_chunk)

# Concatenate all processed chunks along the time dimension (axis=1)
audio_out = torch.cat(processed_chunks, dim=1)

torchaudio.save(
    audio_outpath, audio_out, sample_rate=sr, encoding="PCM_S", bits_per_sample=16
)
