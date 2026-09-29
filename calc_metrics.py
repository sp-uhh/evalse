from os.path import join
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import librosa

from pesq import pesq
from pystoi import stoi

import torchaudio

# scoreq[gpu]
from scoreq import Scoreq
import distillmos
import nemo.collections.asr as nemo_asr
import logging as _logging
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(nemo_logging.ERROR)
for _handler in _logging.getLogger("nemo_logger").handlers:
    _handler.addFilter(lambda record: record.levelno >= _logging.ERROR)

import gc
import re
import jiwer
import jiwer.transforms as tr
from num2words import num2words

class Num2String(tr.AbstractTransform):
    """Transform to normalize text, converting numbers inside strings into words"""

    def _conv_num(self, match):
        return num2words(match.group())

    def process_string(self, s: str):
        return re.sub(r'\b\d+\b', self._conv_num, s)

class GonnaWanna(tr.AbstractTransform):

    def process_string(self, s: str):
        s = re.sub(r"wanna", "want to", s)
        s = re.sub(r"gonna", "going to", s)
        return s

class SplitUnits(tr.AbstractTransform):

    def process_string(self, s: str):
        return re.sub(r'(\d[\.\d]*)', r'\1 ', s)


WER_TFS = jiwer.Compose(
    [
        SplitUnits(),
        Num2String(),
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        GonnaWanna(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def si_sdr_components(s_hat, s, n):
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise

    return s_target, e_noise, e_art

def si_sdr(s_hat, s, n):
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)
    return 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)

def mean_std(data):
    total = len(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    notnan = len(data)
    return mean, std, notnan, total

def safe_wacc(ref, hyp):
    """Word accuracy = 1 - WER (NaN if the reference is empty)"""
    if not ref or not ref.strip():
        return np.nan
    return 1 - jiwer.wer(ref, hyp, reference_transform=WER_TFS, hypothesis_transform=WER_TFS)

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--text_dir", type=str, default=None, required=False, help='Directory containing transcriptions')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [],
            "distillmos": [], "scoreq_nr": [], "scoreq_ref": [],
            "wacc_quartznet": [], "wacc_pkctc0b6": []}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build file list
    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))

    # Precompute file paths
    file_info = []
    for noisy_file in noisy_files:
        filename = noisy_file.replace(args.noisy_dir, "").lstrip('/')
        if 'dB' in filename: # EARS-WHAM format
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        clean_filepath = join(args.clean_dir, clean_filename)
        noisy_filepath = join(args.noisy_dir, filename)
        enhanced_filepath = join(args.enhanced_dir, filename)
        file_info.append((filename, clean_filepath, noisy_filepath, enhanced_filepath))

    n_files = len(file_info)

    # ── Phase 1: PESQ + ESTOI + SI-SDR + ScoreQ (GPU ONNX) + DistillMOS (GPU) ──
    print(f"=== Phase 1: PESQ + ESTOI + SI-SDR + ScoreQ + DistillMOS ({n_files} files) ===")

    scoreq_nr = Scoreq(data_domain='natural', mode='nr')
    scoreq_ref = Scoreq(data_domain='natural', mode='ref')

    sqa_model = distillmos.ConvTransformerSQAModel().to(device)
    sqa_model.eval()

    for filename, clean_filepath, noisy_filepath, enhanced_filepath in tqdm(file_info, desc="Phase 1"):
        x, sr_x = read(clean_filepath)
        y, sr_y = read(noisy_filepath)
        x_hat, sr_x_hat = read(enhanced_filepath)
        assert sr_x == sr_y == sr_x_hat

        # DistillMOS on the full (untrimmed) enhanced signal
        x_hat_t = torch.from_numpy(x_hat).float().unsqueeze(0)
        x_hat_t_16k = torchaudio.transforms.Resample(sr_x_hat, 16000)(x_hat_t) if sr_x_hat != 16000 else x_hat_t

        if len(x_hat) != len(x):
            length = min(len(x_hat), len(x))
            x_hat = x_hat[0:length] + np.spacing(1)
            x = x[0:length] + np.spacing(1)
            y = y[0:length] + np.spacing(1)

        n = y - x
        x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x

        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        data["si_sdr"].append(si_sdr(x_hat, x, n))

        data["scoreq_nr"].append(scoreq_nr.predict(test_path=enhanced_filepath))
        data["scoreq_ref"].append(scoreq_ref.predict(test_path=enhanced_filepath, ref_path=clean_filepath))

        with torch.no_grad():
            data["distillmos"].append(sqa_model(x_hat_t_16k.to(device)).item())

    del scoreq_nr, scoreq_ref, sqa_model
    free_gpu()

    # ── Phase 2: QuartzNet (GPU) + Parakeet CTC 0.6B (GPU) ──
    print(f"=== Phase 2: QuartzNet + Parakeet CTC 0.6B ({n_files} files) ===")

    asr_model_qn = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En").to(device)
    asr_model_pk = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b").to(device)

    already_printed = False
    for filename, clean_filepath, noisy_filepath, enhanced_filepath in tqdm(file_info, desc="Phase 2"):
        transcription_qn = asr_model_qn.transcribe([enhanced_filepath], verbose=False)[0].text
        transcription_pk = asr_model_pk.transcribe([enhanced_filepath], verbose=False)[0].text
        if args.text_dir is not None:
            text_filepath = join(args.text_dir, filename.replace('.wav', '.txt'))
            with open(text_filepath, 'r') as file:
                ref_transcription = file.read().rstrip()
                ref_transcription_qn = ref_transcription
                ref_transcription_pk = ref_transcription
        else:
            if not already_printed:
                print(f"No text file provided, transcribing {clean_filepath} and others...")
                already_printed = True
            ref_transcription_qn = asr_model_qn.transcribe([clean_filepath], verbose=False)[0].text
            ref_transcription_pk = asr_model_pk.transcribe([clean_filepath], verbose=False)[0].text

        data["wacc_quartznet"].append(safe_wacc(ref_transcription_qn, transcription_qn))
        data["wacc_pkctc0b6"].append(safe_wacc(ref_transcription_pk, transcription_pk))

    del asr_model_qn, asr_model_pk
    free_gpu()

    df = pd.DataFrame(data)

    # Summary lines (WAcc reported in percent, i.e. 100 - WER[%])
    lines = [
        "PESQ: {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(df["pesq"].to_numpy())),
        "ESTOI: {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(df["estoi"].to_numpy())),
        "SI-SDR: {:.1f} ± {:.1f} ({:d}/{:d})".format(*mean_std(df["si_sdr"].to_numpy())),
        "DistillMOS: {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(df["distillmos"].to_numpy())),
        "SCOREQ (NR): {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(df["scoreq_nr"].to_numpy())),
        "SCOREQ (REF): {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(df["scoreq_ref"].to_numpy())),
        "WAcc (QuartzNet): {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(100*df["wacc_quartznet"].to_numpy())),
        "WAcc (Parakeet CTC 0.6B): {:.2f} ± {:.2f} ({:d}/{:d})".format(*mean_std(100*df["wacc_pkctc0b6"].to_numpy())),
    ]

    for line in lines:
        print(line)

    # Save average results to file
    with open(join(args.enhanced_dir, "_avg_results.txt"), "w") as log:
        log.write("\n".join(lines) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)
