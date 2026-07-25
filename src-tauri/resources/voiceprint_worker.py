#!/usr/bin/env python3
"""Voiceprint worker for YuMo (语墨) — speaker verification one-shot process.

Protocol (matches custom_model_worker.py):
    stdin  — single JSON command line
    stdout — last line JSON: {"ok": true, ...} or {"ok": false, "error": "..."}
    stderr — log lines
    exit   — 0 on success, non-zero on failure
    cwd    — ~/.voiceink (set by Rust caller)

Actions:
    ensure_models     — download CAM++ ONNX to voiceprint/models/
    enroll_from_history — cluster recordings → speaker profile.npy
    filter            — VAD → embed → cosine gate → output filtered WAV
    status            — enrolled? models ready?
    clear             — remove profile.npy / profile.json
"""

import json
import os
import sys
import struct
import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMPPUS_ONNX_URL = (
    "https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx/resolve/main/"
    "campplus_cn_en_common_200k.onnx"
)

SAMPLE_RATE = 16000
FBANK_DIM = 80
EMBED_DIM = 192  # CAM++ 3DSpeaker 200k variant

VAD_SR = 16000
VAD_WIN_SAMPLES = 512       # 32ms at 16kHz
VAD_MS_PER_FRAME = 32
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 100
VAD_PAD_MS = 30
VAD_THRESHOLD = 0.5

FADE_MS = 10
FADE_SAMPLES = int(FADE_MS / 1000.0 * SAMPLE_RATE)

WORKER_DIR = Path(os.getcwd())  # cwd is ~/.voiceink
VOICEPRINT_DIR = WORKER_DIR / "voiceprint"
MODELS_DIR = VOICEPRINT_DIR / "models"
PROFILE_NPY = VOICEPRINT_DIR / "profile.npy"
PROFILE_JSON = VOICEPRINT_DIR / "profile.json"
CAMPPUS_ONNX = MODELS_DIR / "campplus_cn.onnx"
META_JSON = MODELS_DIR / "campplus_cn.meta.json"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[voiceprint] {msg}", file=sys.stderr, flush=True)


def respond(data: dict[str, Any]) -> None:
    line = json.dumps(data, ensure_ascii=False)
    print(line, flush=True)


def ok_response(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, **kwargs}


def err_response(msg: str) -> dict[str, Any]:
    return {"ok": False, "error": msg}


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read WAV as float32 mono, returns (samples, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def write_wav(path: str | Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf
    sf.write(str(path), samples, sr, subtype="PCM_16")


def resample_16k(samples: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SAMPLE_RATE:
        return samples.copy()
    import librosa
    return librosa.resample(samples, orig_sr=src_sr, target_sr=SAMPLE_RATE)


def compute_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> None:
    import urllib.request
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    log(f"downloading {url} → {dest.name} ...")
    try:
        with urllib.request.urlopen(url) as resp:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        log(f"    done ({size_mb:.1f} MB)")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# VAD — Silero VAD via torch.hub (reliable, avoids ONNX export issues)
# ---------------------------------------------------------------------------

_vad_model = None
_vad_utils = None


def _get_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    return _vad_model, _vad_utils


def load_vad():
    """Return the Silero VAD model (torch module)."""
    model, _ = _get_vad()
    return model


def vad_segments(
    session,  # kept for compatibility — actually a torch module
    audio: np.ndarray,
    min_speech_ms: int = VAD_MIN_SPEECH_MS,
    min_silence_ms: int = VAD_MIN_SILENCE_MS,
    pad_ms: int = VAD_PAD_MS,
) -> list[tuple[int, int]]:
    """Returns list of (start_sample, end_sample) speech segments."""
    if len(audio) == 0:
        return []

    _, utils = _get_vad()
    get_speech_timestamps = utils[0]

    # Silero VAD expects float32 tensor at 16kHz
    audio_t = torch.from_numpy(audio.astype(np.float32))

    timestamps = get_speech_timestamps(
        audio_t,
        session,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )

    segments = [(ts["start"], ts["end"]) for ts in timestamps]
    return segments


# ---------------------------------------------------------------------------


def extract_fbank(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract 80-dim log Mel filterbank features.

    Returns (num_frames, FBANK_DIM) float32 array.
    """
    # Pre-emphasis
    preemph = 0.97
    samples = np.append(samples[0:1], samples[1:] - preemph * samples[:-1])

    # Framing: 25ms window, 10ms shift
    frame_len = int(0.025 * sr)
    frame_shift = int(0.010 * sr)
    n_frames = max(1, (len(samples) - frame_len) // frame_shift + 1)

    window = np.hamming(frame_len)

    frames = np.zeros((n_frames, frame_len), dtype=np.float64)
    for i in range(n_frames):
        start = i * frame_shift
        frames[i] = samples[start : start + frame_len] * window

    # Power spectrum
    n_fft = 512
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2

    # Mel filterbank
    n_mels = FBANK_DIM
    low_freq = 20
    high_freq = sr // 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bin_points[m], bin_points[m + 1]):
            fbank[m, k] = (k - bin_points[m]) / (bin_points[m + 1] - bin_points[m])
        for k in range(bin_points[m + 1], bin_points[m + 2]):
            fbank[m, k] = (bin_points[m + 2] - k) / (bin_points[m + 2] - bin_points[m + 1])
    feat = np.log(np.dot(spec, fbank.T) + 1e-20)

    # CMVN per utterance
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-10
    feat = (feat - mean) / std

    return feat.astype(np.float32)
# ---------------------------------------------------------------------------
# CAM++ Embedder
# ---------------------------------------------------------------------------

def load_embedder() -> ort.InferenceSession:
    if not CAMPPUS_ONNX.exists():
        raise FileNotFoundError(f"Embedder model not found: {CAMPPUS_ONNX}")
    return ort.InferenceSession(
        str(CAMPPUS_ONNX),
        providers=["CPUExecutionProvider"],
        sess_options=_ort_session_options(),
    )


def _ort_session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def extract_embedding(
    session: ort.InferenceSession,
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract L2-normalized speaker embedding from audio."""
    if sr != SAMPLE_RATE:
        audio = resample_16k(audio, sr)

    # Extract fbank features
    fbank = extract_fbank(audio)
    if fbank.shape[0] < 10:
        # Short audio — pad to minimum length
        pad_len = 10 - fbank.shape[0]
        fbank = np.pad(fbank, ((0, pad_len), (0, 0)), mode="edge")

    # Run model — input name varies by ONNX export
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: fbank[np.newaxis, :, :]}
    outputs = session.run(None, ort_inputs)

    # Output is (1, EMBED_DIM) or (1, 1, EMBED_DIM) depending on export
    emb = np.array(outputs[0]).squeeze()
    if emb.ndim > 1:
        emb = emb[-1]  # take last frame if time-pooled

    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_ensure_models() -> dict[str, Any]:
    # Download CAM++ ONNX if missing
    if not CAMPPUS_ONNX.exists():
        log("downloading CAM++ speaker embedding model...")
        download_file(CAMPPUS_ONNX_URL, CAMPPUS_ONNX)

    # Verify torch has Silero VAD (downloaded via torch.hub on first use)
    try:
        _get_vad()
        log("Silero VAD ready (torch.hub)")
    except Exception as e:
        return err_response(f"Silero VAD not available: {e}")

    # Dry-run: verify CAM++ loads and get embedding dim
    sess = load_embedder()
    dry_input = np.zeros((10, FBANK_DIM), dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    dry_out = sess.run(None, {input_name: dry_input[np.newaxis, :, :]})
    dim = int(np.array(dry_out[0]).squeeze().shape[-1]) if np.array(dry_out[0]).ndim > 1 else int(np.array(dry_out[0]).shape[0])

    # Write meta
    META_JSON.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_url": CAMPPUS_ONNX_URL,
        "sha256": _file_sha256(CAMPPUS_ONNX),
        "dim": dim,
        "model_id": "campplus_3dspeaker_200k_cn_en",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    return ok_response(
        embedder=str(CAMPPUS_ONNX),
        dim=dim,
        vad="torch.hub:snakers4/silero-vad",
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def action_enroll_from_history(
    recordings_dir: str,
    max_files: int = 800,
    min_sec: float = 1.5,
    max_sec: float = 45.0,
    min_rms: float = 0.008,
) -> dict[str, Any]:
    rec_dir = Path(recordings_dir)
    if not rec_dir.is_dir():
        return err_response(f"Recordings directory not found: {rec_dir}")

    # Load models
    vad_sess = load_vad()
    emb_sess = load_embedder()

    # Collect files: prefer recording_*.wav
    wavs = sorted(rec_dir.glob("recording_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(wavs) < max_files:
        other_wavs = sorted(
            [p for p in rec_dir.glob("*.wav") if not p.name.startswith("recording_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        remaining = max_files - len(wavs)
        wavs.extend(other_wavs[:remaining])

    log(f"found {len(wavs)} candidate files (max_files={max_files})")

    embeddings: list[np.ndarray] = []
    used_files = 0

    for i, wav_path in enumerate(wavs):
        if len(embeddings) >= max_files:
            break

        try:
            samples, sr = read_wav(wav_path)
            dur = len(samples) / sr if sr > 0 else 0

            if dur < min_sec or dur > max_sec:
                continue

            if sr != SAMPLE_RATE:
                samples = resample_16k(samples, sr)
                sr = SAMPLE_RATE

            rms = compute_rms(samples)
            if rms < min_rms:
                continue

            # VAD → take all speech frames merged
            segs = vad_segments(vad_sess, samples)
            if not segs:
                continue

            # Merge speech into one contiguous array
            speech_parts = [samples[s:e] for s, e in segs]
            speech = np.concatenate(speech_parts) if speech_parts else samples

            emb = extract_embedding(emb_sess, speech, sr)
            embeddings.append(emb)
            used_files += 1

            if (i + 1) % 20 == 0:
                log(f"progress enrolled={used_files}/{i+1}")

        except Exception as e:
            log(f"skip {wav_path.name}: {e}")
            continue

    log(f"extracted {len(embeddings)} embeddings from {used_files} files")

    if len(embeddings) < 15:
        return err_response(
            "历史录音不够：仅有 {} 段有效语音。请积累更多本人独说录音后再试（至少需要 15 段）。".format(
                len(embeddings)
            )
        )

    # Clustering via AgglomerativeClustering (cosine distance)
    from sklearn.cluster import AgglomerativeClustering

    emb_matrix = np.stack(embeddings, axis=0)
    cluster = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        n_clusters=None,
        distance_threshold=0.35,
    )
    labels = cluster.fit_predict(emb_matrix)

    # Find largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    cluster_mask = labels == largest_cluster
    cluster_size = int(cluster_mask.sum())

    min_cluster = max(15, int(0.05 * len(embeddings)))
    if cluster_size < min_cluster:
        return err_response(
            "历史录音说话人不够一致：最大簇仅 {} 段，占 {:.0f}%。"
            "请积累更多本人独说录音后再试。".format(
                cluster_size, cluster_size / len(embeddings) * 100
            )
        )

    # Profile = mean of cluster embeddings, L2 normalized
    profile = emb_matrix[cluster_mask].mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    # Save
    VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROFILE_NPY, profile.astype(np.float32))

    meta = _read_meta()
    profile_meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": meta.get("model_id", "unknown"),
        "dim": int(profile.shape[0]),
        "n_files_used": used_files,
        "n_embeddings": len(embeddings),
        "cluster_size": cluster_size,
        "distance_threshold": 0.35,
    }
    with open(PROFILE_JSON, "w") as f:
        json.dump(profile_meta, f, indent=2)

    log(f"profile saved: dim={profile.shape[0]} cluster={cluster_size}/{len(embeddings)}")

    return ok_response(
        dim=int(profile.shape[0]),
        n_files_used=used_files,
        n_embeddings=len(embeddings),
        cluster_size=cluster_size,
        profile_path=str(PROFILE_NPY),
    )


def action_filter(
    audio: str = "",
    audio_path: str = "",
    output: str = "",
    threshold: float = 0.62,
) -> dict[str, Any]:
    path = audio or audio_path
    if not PROFILE_NPY.exists():
        return err_response("profile_missing")

    profile = np.load(PROFILE_NPY).astype(np.float32)

    vad_sess = load_vad()
    emb_sess = load_embedder()

    samples, sr = read_wav(path)
    if sr != SAMPLE_RATE:
        samples = resample_16k(samples, sr)
        sr = SAMPLE_RATE

    output_samples = samples.copy()

    segs = vad_segments(vad_sess, samples)
    if not segs:
        # No speech detected — output silence
        write_wav(output, np.zeros_like(samples))
        return ok_response(kept_ratio=0.0, n_segments=0, n_kept=0, mean_sim_kept=0.0, mean_sim_rejected=0.0)

    kept_segments = 0
    total_speech_samples = 0
    kept_samples = 0
    sims_kept: list[float] = []
    sims_rejected: list[float] = []

    for seg_start, seg_end in segs:
        seg_len = seg_end - seg_start
        total_speech_samples += seg_len
        seg_audio = samples[seg_start:seg_end]

        emb = extract_embedding(emb_sess, seg_audio, SAMPLE_RATE)
        sim = float(np.dot(emb, profile))

        if sim >= threshold:
            kept_segments += 1
            kept_samples += seg_len
            sims_kept.append(sim)
        else:
            # Zero out non-matching speech
            output_samples[seg_start:seg_end] = 0.0
            # Apply 10ms fade at boundaries to avoid clicks
            _apply_fade(output_samples, seg_start, seg_end)
            sims_rejected.append(sim)

    # Compute kept_ratio
    kept_ratio = kept_samples / total_speech_samples if total_speech_samples > 0 else 0.0
    mean_sim_kept = float(np.mean(sims_kept)) if sims_kept else 0.0
    mean_sim_rejected = float(np.mean(sims_rejected)) if sims_rejected else 0.0

    write_wav(output, output_samples)
    return ok_response(
        kept_ratio=round(kept_ratio, 4),
        n_segments=len(segs),
        n_kept=kept_segments,
        mean_sim_kept=round(mean_sim_kept, 4),
        mean_sim_rejected=round(mean_sim_rejected, 4),
    )


def _apply_fade(samples: np.ndarray, start: int, end: int) -> None:
    """Linear 10ms fade-in/out at segment boundaries to avoid clicks."""
    fade_len = min(FADE_SAMPLES, (end - start) // 2)
    if fade_len <= 0:
        return

    # Fade out at end
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    if end - fade_len >= 0:
        samples[end - fade_len : end] *= fade_out

    # Fade in at start
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    if start + fade_len <= len(samples):
        samples[start : start + fade_len] *= fade_in


def _read_meta() -> dict[str, Any]:
    if META_JSON.exists():
        with open(META_JSON) as f:
            return json.load(f)
    return {}


def _read_profile_meta() -> dict[str, Any] | None:
    if PROFILE_JSON.exists():
        with open(PROFILE_JSON) as f:
            return json.load(f)
    return None

def action_status() -> dict[str, Any]:
    enrolled = PROFILE_NPY.exists()
    models_ready = CAMPPUS_ONNX.exists()
    meta = _read_meta()
    profile_meta = _read_profile_meta()

    return ok_response(
        enrolled=enrolled,
        dim=meta.get("dim"),
        profile=profile_meta,
        models_ready=models_ready,
    )


def action_clear() -> dict[str, Any]:
    for p in [PROFILE_NPY, PROFILE_JSON]:
        if p.exists():
            p.unlink()
            log(f"removed {p.name}")
    return ok_response()


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "ensure_models": action_ensure_models,
    "enroll_from_history": action_enroll_from_history,
    "filter": action_filter,
    "status": action_status,
    "clear": action_clear,
}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        log("dry-run: checking imports...")
        import sklearn  # noqa: F401
        import scipy  # noqa: F401
        import soundfile  # noqa: F401
        import librosa  # noqa: F401
        log("dry-run ok")
        return

    try:
        raw = sys.stdin.readline()
        if not raw:
            respond(err_response("empty stdin"))
            sys.exit(1)

        cmd = json.loads(raw)
    except json.JSONDecodeError as e:
        respond(err_response(f"invalid JSON: {e}"))
        sys.exit(1)
    except Exception as e:
        respond(err_response(f"read stdin: {e}"))
        sys.exit(1)

    action = cmd.get("action", "")
    if action not in ACTION_MAP:
        respond(err_response(f"unknown action: {action}"))
        sys.exit(1)

    try:
        fn = ACTION_MAP[action]
        kwargs = {k: v for k, v in cmd.items() if k != "action"}
        result = fn(**kwargs)
        respond(result)
        if not result.get("ok", False):
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        respond(err_response(f"{type(e).__name__}: {e}"))
        sys.exit(1)


if __name__ == "__main__":
    main()
