# evalse — Speech enhancement evaluation metrics

Evaluation script accompanying the paper

> **Perceptual Quality Loss or Loss of Perceptual Quality?**
> Danilo de Oliveira, Tal Peer, Maurício do V. M. da Costa, Timo Gerkmann

`calc_metrics.py` computes the following metrics for a directory of enhanced speech files:

| Metric | Type | Model / implementation | Better |
|---|---|---|---|
| [PESQ](https://doi.org/10.1109/ICASSP.2001.941023) (wide-band) | intrusive | [`pesq`](https://github.com/ludlows/PESQ) | ↑ |
| [ESTOI](https://doi.org/10.1109/TASLP.2016.2585878) | intrusive | [`pystoi`](https://github.com/mpariente/pystoi) | ↑ |
| [SI-SDR](https://doi.org/10.1109/ICASSP.2019.8683855) [dB] | intrusive | scale-invariant SDR | ↑ |
| [DistillMOS](https://doi.org/10.1109/ICASSP49660.2025.10888007) | non-intrusive | [`distillmos`](https://github.com/microsoft/Distill-MOS) | ↑ |
| [SCOREQ](https://doi.org/10.52202/079017-3353) (NR) | non-intrusive | [`scoreq`](https://github.com/alessandroragano/scoreq), natural-speech domain | ↑ |
| [SCOREQ](https://doi.org/10.52202/079017-3353) (REF) | intrusive | [`scoreq`](https://github.com/alessandroragano/scoreq), natural-speech domain (distance to reference) | ↓ |
| WAcc [QuartzNet 15x5](https://doi.org/10.1109/ICASSP40776.2020.9053889) [%] | content-intrusive | NVIDIA NeMo `QuartzNet15x5Base-En` | ↑ |
| WAcc [Parakeet CTC 0.6B](https://doi.org/10.1109/ASRU57964.2023.10389701) [%] | content-intrusive | NVIDIA NeMo `nvidia/parakeet-ctc-0.6b` | ↑ |

Word accuracy is defined as WAcc = 1 − WER (reported as 100 − WER in %). Since WER can exceed 100 %, WAcc can be negative.

## Installation

Tested with Python 3.11 on Linux with an NVIDIA GPU. Everything also runs on CPU, but much more slowly.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# scoreq depends on the CPU build of onnxruntime, which shadows onnxruntime-gpu.
# Remove it and reinstall the GPU build so SCOREQ runs on CUDA:
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.30.0
```

When the script starts, it should print `SCOREQ (ONNX) initialized on provider: CUDAExecutionProvider`.

### FFmpeg

Recent versions of `torchaudio` load audio through `torchcodec`, which needs the FFmpeg shared libraries (`libavutil`, `libavcodec`, ...). SCOREQ reads audio through `torchaudio.load`, so you need FFmpeg installed, e.g.

```bash
sudo apt install ffmpeg        # Debian/Ubuntu
# or
conda install -c conda-forge ffmpeg
```

If FFmpeg is installed in a non-standard location (e.g. an environment module on an HPC cluster), add its `lib` directory to the library path before running the script:

```bash
export LD_LIBRARY_PATH=/path/to/ffmpeg/lib:$LD_LIBRARY_PATH
```

### NeMo version

`nemo_toolkit` is pinned to 2.7.3 because NeMo 3.x no longer provides the `QuartzNet15x5Base-En` checkpoint.

Pretrained models (SCOREQ, DistillMOS, QuartzNet, Parakeet) are downloaded automatically on first use.

## Usage

```bash
python calc_metrics.py \
    --clean_dir    /path/to/clean \
    --noisy_dir    /path/to/noisy \
    --enhanced_dir /path/to/enhanced \
    [--text_dir    /path/to/transcripts]
```

- **Directory layout.** The list of files to evaluate is taken from `--noisy_dir` (all `*.wav` files, including subdirectories). Each enhanced file must have the same relative path under `--enhanced_dir`. The clean reference has the same relative path under `--clean_dir`. The exception is noisy filenames containing `dB`, in which case the clean filename is the part before the first `_` plus `.wav` (e.g. `p102/00181_-0.1dB.wav` → `p102/00181.wav`, as in EARS-WHAM).
- **Sampling rate.** Clean, noisy and enhanced files must have the same sampling rate. PESQ and DistillMOS are computed on 16 kHz versions of the signals (resampled if needed). ESTOI and SI-SDR use the original rate. If the enhanced and clean signals differ in length, both are trimmed to the shorter length for the intrusive metrics.
- **Reference transcriptions.** If `--text_dir` is given, the reference text for `<name>.wav` is read from `<text_dir>/<name>.txt`. Otherwise, each ASR model transcribes the clean file, and that output is used as the reference. The WAcc then measures how well the model's own transcription of the clean speech is preserved.
- **Text normalization.** Before WER is computed, transcripts are lower-cased, numbers are converted to words, English contractions are expanded, and punctuation and extra whitespace are removed (see `WER_TFS` in the script).

### Output

The results are printed and written to `--enhanced_dir`:

- `_results.csv`: per-file scores (WAcc as a fraction in [−∞, 1])
- `_avg_results.txt`: mean ± standard deviation over all files (WAcc in %), followed by `(valid/total)` file counts

## License

This code is released under the [MIT License](LICENSE). The metric packages and pretrained models it uses are distributed under their own licenses.
