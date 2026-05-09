import io
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import streamlit as st
from huggingface_hub import hf_hub_download
from model import Wav2Vec2Model

REPO_ID    = "eunei/ai-cover-detector-singgraph"
MODEL_FILE = "best.pth"
SR_XLSR   = 16000
SR_MERT   = 24000
CUT_XLSR   = 64600                        # 학습 시 사용한 샘플 수
CUT_MERT   = int(64600 * 24000 / 16000)
THRESHOLD = 0.5


@st.cache_resource(show_spinner="모델 가중치 로딩 중...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

    class _Args: pass
    model = Wav2Vec2Model(_Args(), device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model.eval().to(device)
    return model, device


def load_and_clip(file_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(file_bytes))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    wav_16k = torchaudio.transforms.Resample(sr, SR_XLSR)(waveform)
    wav_24k = torchaudio.transforms.Resample(sr, SR_MERT)(waveform)

    clip_16k = CUT_XLSR   # 64640
    clip_24k = CUT_MERT   # 96960
    ratio    = SR_MERT / SR_XLSR         # 1.5
    total    = wav_16k.shape[1]

    clips_xlsr, clips_mert, timestamps = [], [], []
    for start in range(0, total - clip_16k + 1, clip_16k):
        end_16k   = start + clip_16k
        start_24k = int(start * ratio)
        end_24k   = start_24k + clip_24k
        if end_24k > wav_24k.shape[1]:
            break
        clips_xlsr.append(wav_16k[:, start:end_16k])
        clips_mert.append(wav_24k[:, start_24k:end_24k])
        t_s = start // SR_XLSR
        t_e = t_s + int(CLIP_SEC)
        timestamps.append(f"{t_s//60:02d}:{t_s%60:02d} ~ {t_e//60:02d}:{t_e%60:02d}")

    return clips_xlsr, clips_mert, timestamps


def run_inference(uploaded_file) -> dict:
    model, device = load_model()
    clips_xlsr, clips_mert, timestamps = load_and_clip(uploaded_file.read())

    if not clips_xlsr:
        return {"label":"분석 불가","score":0,
                "summary":"음원이 너무 짧습니다 (최소 4초 이상 필요).","segments":[]}

    scores, suspicious = [], []
    with torch.no_grad():
        for x, x2, ts in zip(clips_xlsr, clips_mert, timestamps):
            logits = model(x.unsqueeze(-1).to(device), x2.unsqueeze(-1).to(device))
            prob   = F.softmax(logits, dim=-1)[0, 1].item()
            scores.append(prob)
            if prob >= THRESHOLD:
                suspicious.append({"time": ts, "score": round(prob, 2)})

    top_k      = max(1, len(scores) // 3)
    song_score = float(np.mean(sorted(scores, reverse=True)[:top_k]))
    song_pct   = round(song_score * 100)
    is_ai      = song_score >= THRESHOLD

    return {
        "label":    "AI 커버곡 가능성 높음" if is_ai else "AI 커버곡 가능성 낮음",
        "score":    song_pct,
        "summary":  (f"분석된 {len(clips_xlsr)}개 구간 중 {len(suspicious)}개 구간에서 AI 보컬 패턴이 감지되었습니다."
                     if suspicious else
                     f"분석된 {len(clips_xlsr)}개 구간 모두에서 AI 보컬 패턴이 감지되지 않았습니다."),
        "segments": suspicious[:5],
    }
