# NeuCodec 🎧

HuggingFace 🤗: [Model](https://huggingface.co/neuphonic/neucodec), [Distilled Model](https://huggingface.co/neuphonic/distill-neucodec)


[NeuCodec Demo](https://github.com/user-attachments/assets/c03745cd-a8c8-46ca-8f5d-ba3af091923f)

*Created by Neuphonic - building faster, smaller, on-device voice AI*

A lightweight neural codec that encodes audio at just 0.8 kbps - perfect for researchers and builders who need something that *just works* for training high quality text-to-speech models.

# Key Features

🔊 Low bit-rate compression - a speech codec that compresses and reconstructs audio with near-inaudible reconstruction loss
<br>
🎼 Upsamples from 16kHz → 24kHz
<br>
🌍 Ready for real-world use - train your own SpeechLMs without needing to build your own codec
<br>
🏢 Commercial use permitted - use it in your own tools or products
<br>
📊 Released with large pre-encoded datasets - we’ve compressed Emilia-YODAS from 1.7TB to 41GB using NeuCodec, significantly reducing the compute requirements needed for training 
<br>

# Model Details

NeuCodec is a Finite Scalar Quantisation (FSQ) based 0.8kbps audio codec for speech tokenization.
It takes advantage of the following features:

* It uses both audio (BigCodec) and semantic ([Wav2Vec2-BERT](https://huggingface.co/facebook/w2v-bert-2.0)) encoders. 
* We make use of Finite Scalar Quantisation (FSQ) resulting in a single vector for the quantised output, which makes it ideal for downstream modeling with Speech Language Models.
* At 50 tokens/sec and 16 bits per token, the overall bit-rate is 0.8kbps.
* The codec takes in 16kHz input and outputs 24kHz using an upsampling decoder.

Our work largely based on extending the work of [X-Codec2.0](https://huggingface.co/HKUSTAudio/xcodec2).

- **Developed by:** Neuphonic
- **Model type:** Neural Audio Codec
- **License:** apache-2.0
- **Repository:** https://github.com/neuphonic/neucodec
- **Paper:** [arXiv](https://arxiv.org/abs/2509.09550)
- **Pre-encoded Datasets**:
  - [Emilia-YODAS-EN](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec)

## Get Started

Use the code below to get started with the model.

To install from pypi in a dedicated environment:

**Using conda + pip:**
```bash
conda create -n neucodec python>3.9
conda activate neucodec
pip install neucodec
```

**Using uv:**
```bash
uv venv neucodec --python 3.10
source neucodec/bin/activate  # On Windows: neucodec\Scripts\activate
uv pip install neucodec
```

If you would like to use the onnx decoder, also install `onnxruntime`:
```bash
pip install onnxruntime
```
Then, to use the regular codec in python:

```python
import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from neucodec import NeuCodec
 
model = NeuCodec.from_pretrained("neuphonic/neucodec")
model.eval().cuda()   
 
y, sr = torchaudio.load(librosa.ex("libri1"))
if sr != 16_000:
    y = T.Resample(sr, 16_000)(y)[None, ...] # (B, 1, T_16)

with torch.no_grad():
    fsq_codes = model.encode_code(y)
    # fsq_codes = model.encode_code(librosa.ex("libri1")) # or directly pass your filepath!
    print(f"Codes shape: {fsq_codes.shape}")  
    recon = model.decode_code(fsq_codes).cpu() # (B, 1, T_24)

torchaudio.save("reconstructed.wav", recon[0, :, :], 24_000)
```

To run in fp16 for faster inference (decode is ~1.64× faster with no accuracy loss):

```python
model = NeuCodec.from_pretrained("neuphonic/neucodec")
model.eval().cuda().half()  # convert to fp16; ISTFT and FSQ quantizer stay in fp32 automatically

with torch.no_grad():
    fsq_codes = model.encode_code(y)
    recon = model.decode_code(fsq_codes).cpu()  # (B, 1, T_24), always float32

torchaudio.save("reconstructed.wav", recon[0, :, :], 24_000)
```

## Benchmarks

Speed and accuracy benchmarks compare `neucodec==0.0.4` (baseline) against the current release running in fp32 and fp16. `neucodec==0.0.4` is automatically installed to `/tmp` on the first run — no manual setup needed for the baseline.

### Setup

Install [UTMOSv2](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2) for neural MOS quality scoring:

```bash
pip3 install git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2
```

### Accuracy benchmark

Compares UTMOS (predicted MOS), MSE and SNR vs the original waveform across three variants: `v0.0.4_fp32`, `current_fp32`, `current_fp16`.

```bash
python3 benchmark_accuracy.py \
    --audio_dir Elise_audio \
    --num_files 100 \
    --num_repetitions 20 \
    --out results.csv
```

| Argument | Default | Description |
|---|---|---|
| `--audio_dir` | required | Directory of `.wav`/`.flac`/`.mp3`/`.ogg`/`.opus` files |
| `--num_files` | 100 | Max files sampled from `audio_dir` |
| `--num_repetitions` | 20 | Encode/decode calls averaged per file for timing |
| `--out` | results.csv | Output CSV path |
| `--model` | neuphonic/neucodec | `neuphonic/neucodec` or `neuphonic/distill-neucodec` |
| `--device` | cuda | `cuda` or `cpu` |

### Speed benchmark

Reports encode latency, decode latency, and real-time factor (RTF) per variant with speedup ratios.

```bash
python3 benchmark_speed.py \
    --audio_dir /path/to/wavs \
    --num_files 50 \
    --num_repetitions 20
```

Results on 50 Elise audio files, RTX 3090 Ti, 20 timing repetitions per file:

**Speed**

```
────────────────────────────────────────────────────────────────────────
  Variant           Enc ms    Dec ms    Total ms     RTF  Files
────────────────────────────────────────────────────────────────────────
  v0.0.4_fp32        126.7      11.9       138.7   54.39x     50
  current_fp32       126.3      11.9       138.3   54.56x     50
  current_fp16       119.7       7.3       127.0   60.24x     50
────────────────────────────────────────────────────────────────────────
  speedup v0.0.4 -> current_fp32        x1.00     x1.00       x1.00
  speedup v0.0.4 -> current_fp16        x1.06     x1.64       x1.09
────────────────────────────────────────────────────────────────────────
```

**Accuracy** (MSE and SNR vs original waveform, 50 files, 5 reps)

```
              mse_vs_orig  snr_db_vs_orig  encode_ms  decode_ms
v0.0.4_fp32        0.0022          -1.501      126.6       12.1
current_fp32       0.0022          -1.501      127.1       12.4
current_fp16       0.0022          -1.442      118.3        7.6
```

fp16 decode is **1.64× faster** with identical MSE and a marginal +0.06 dB SNR improvement over fp32. The ISTFT layer always runs in fp32 internally for numerical correctness even when the rest of the model is fp16.

## Training Details

The model was trained using the following data: 
* Emilia-YODAS
* MLS
* LibriTTS
* Fleurs
* CommonVoice
* HUI
* Additional proprietary set

All publically available data was covered by either the CC-BY-4.0 or CC0 license.

## Citation

To cite this project, use the following bibtex entry:

```
@article{julian2025fsq,
  title={Finite Scalar Quantization Enables Redundant and Transmission-Robust Neural Audio Compression at Low Bit-rates},
  author={Julian, Harry and Beeson, Rachel and Konathala, Lohith and Ulin, Johanna and Gao, Jiameng},
  journal={arXiv preprint arXiv:2509.09550},
  year={2025},
  url={https://arxiv.org/abs/2509.09550}
}
```
