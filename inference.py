import io
import os
import json
import math
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import streamlit as st

from huggingface_hub import hf_hub_download
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from model import Wav2Vec2Model


REPO_ID = "eunei/ai-cover-detector-singgraph"
MODEL_FILE = "best.pth"
CONFIG_FILE = "SingGraph.conf"

SR = 16000
CUT = 64600
CLIP_SEC = CUT / SR
THRESHOLD = 0.5

# 네 웹 코드 기준: softmax index 1을 AI score로 사용
AI_INDEX = 1

DEMUCS_MODEL = "mdx_extra"


@st.cache_resource(show_spinner="모델 가중치 로딩 중...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        repo_type="model"
    )

    config_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=CONFIG_FILE,
        repo_type="model"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = Wav2Vec2Model(config["model_config"], device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval().to(device)
    return model, device


@st.cache_resource(show_spinner="VAD 모델 로딩 중...")
def load_vad_pipeline():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN 환경변수가 필요합니다.")

    vad_model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=hf_token,
    )

    pipeline = VoiceActivityDetection(segmentation=vad_model)
    pipeline.instantiate({
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    })

    return pipeline


def save_uploaded_file(uploaded_file, tmp_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix
    input_path = tmp_dir / f"uploaded{suffix}"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    return input_path


def run_demucs(input_path: Path, output_dir: Path):
    cmd = [
        "demucs",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--two-stems=vocals",
        "-n", DEMUCS_MODEL,
        "-o", str(output_dir),
        str(input_path),
    ]

    subprocess.run(cmd, check=True)

    base = output_dir / DEMUCS_MODEL / input_path.stem
    vocals_path = base / "vocals.wav"
    non_vocals_path = base / "no_vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(f"vocals.wav not found: {vocals_path}")

    if not non_vocals_path.exists():
        raise FileNotFoundError(f"no_vocals.wav not found: {non_vocals_path}")

    return vocals_path, non_vocals_path


def parse_pyannote_time(time_str: str) -> float:
    parts = time_str.strip().split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def run_vad(vocals_path: Path):
    pipeline = load_vad_pipeline()

    waveform, sample_rate = sf.read(str(vocals_path), always_2d=True)
    waveform = torch.from_numpy(waveform.T).float()

    vad = pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate,
    })

    segments = []

    for line in str(vad).splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            time_part = line.split("]")[0].split("[")[1]
            start_str, end_str = time_part.split("-->")
            start = parse_pyannote_time(start_str)
            end = parse_pyannote_time(end_str)

            if end > start:
                segments.append((start, end))
        except Exception:
            continue

    return segments


def load_audio_mono(path: Path, sr: int = SR):
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32)


def normalize(x: np.ndarray):
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs
    return x.astype(np.float32)


def fix_length(x: np.ndarray, cut: int = CUT):
    if len(x) >= cut:
        start = (len(x) - cut) // 2
        return x[start:start + cut]

    repeats = int(cut / len(x)) + 1
    return np.tile(x, repeats)[:cut]


def make_clip_pairs(vocals_path: Path, non_vocals_path: Path, segments):
    vocals = load_audio_mono(vocals_path)
    non_vocals = load_audio_mono(non_vocals_path)

    clip_pairs = []
    timestamps = []

    for start_sec, end_sec in segments:
        start = int(start_sec * SR)
        end = int(end_sec * SR)

        if end <= start:
            continue

        vocal_chunk = vocals[start:end]
        non_vocal_chunk = non_vocals[start:end]

        if len(vocal_chunk) == 0 or len(non_vocal_chunk) == 0:
            continue

        vocal_chunk = normalize(fix_length(vocal_chunk))
        non_vocal_chunk = normalize(fix_length(non_vocal_chunk))

        clip_pairs.append((vocal_chunk, non_vocal_chunk))

        t_s = int(start_sec)
        t_e = int(end_sec)
        timestamps.append(
            f"{t_s // 60:02d}:{t_s % 60:02d} ~ {t_e // 60:02d}:{t_e % 60:02d}"
        )

    return clip_pairs, timestamps


def make_bpm_json(non_vocals_path: Path):
    y, sr = librosa.load(str(non_vocals_path), sr=SR, mono=True)

    if len(y) < sr:
        return None

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 4:
        return None

    bpm = int(round(float(np.asarray(tempo).reshape(-1)[0])))

    beat_positions = [int((i % 4) + 1) for i in range(len(beat_times))]
    downbeats = [
        float(beat_times[i])
        for i, pos in enumerate(beat_positions)
        if pos == 1
    ]

    return {
        "bpm": bpm,
        "beat_positions": beat_positions,
        "downbeats": downbeats,
    }


def preprocess_uploaded_audio(uploaded_file):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        input_path = save_uploaded_file(uploaded_file, tmp_dir)

        vocals_path, non_vocals_path = run_demucs(
            input_path=input_path,
            output_dir=tmp_dir / "separated"
        )

        segments = run_vad(vocals_path)

        if len(segments) == 0:
            raise RuntimeError("VAD에서 보컬 구간을 찾지 못했습니다.")

        clip_pairs, timestamps = make_clip_pairs(
            vocals_path=vocals_path,
            non_vocals_path=non_vocals_path,
            segments=segments,
        )

        bpm_info = make_bpm_json(non_vocals_path)

    return clip_pairs, timestamps, bpm_info


def run_inference(uploaded_file) -> dict:
    model, device = load_model()

    try:
        clip_pairs, timestamps, bpm_info = preprocess_uploaded_audio(uploaded_file)
    except Exception as e:
        return {
            "label": "분석 불가",
            "score": 0,
            "summary": f"전처리 실패: {str(e)}",
            "segments": [],
        }

    if not clip_pairs:
        return {
            "label": "분석 불가",
            "score": 0,
            "summary": "유효한 보컬 clip을 생성하지 못했습니다.",
            "segments": [],
        }

    scores = []
    suspicious = []

    with torch.no_grad():
        for (vocal_chunk, non_vocal_chunk), ts in zip(clip_pairs, timestamps):
            x = torch.tensor(vocal_chunk, dtype=torch.float32).unsqueeze(0).to(device)
            x2 = torch.tensor(non_vocal_chunk, dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(x, x2)
            prob = F.softmax(logits, dim=-1)[0, AI_INDEX].item()

            scores.append(prob)

            if prob >= THRESHOLD:
                suspicious.append({
                    "time": ts,
                    "score": round(prob, 2),
                })

    top_k = max(1, len(scores) // 3)
    song_score = float(np.mean(sorted(scores, reverse=True)[:top_k]))
    song_pct = round(song_score * 100)
    is_ai = song_score >= THRESHOLD

    return {
        "label": "AI 커버곡 가능성 높음" if is_ai else "AI 커버곡 가능성 낮음",
        "score": song_pct,
        "summary": (
            f"분석된 {len(clip_pairs)}개 보컬 구간 중 "
            f"{len(suspicious)}개 구간에서 AI 보컬 패턴이 감지되었습니다."
            if suspicious else
            f"분석된 {len(clip_pairs)}개 보컬 구간 모두에서 AI 보컬 패턴이 강하게 감지되지 않았습니다."
        ),
        "segments": suspicious[:5],
        "num_clips": len(clip_pairs),
        "bpm_info": bpm_info,
    }
