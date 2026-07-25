"""Mano-ASR local plugin for VoiceInk (语墨).

This plugin loads Mano-ASR's AutoModel directly in the daemon process —
no external HTTP service, no child process to manage. When the daemon
exits, the model object is garbage-collected and all resources are freed.

Dependencies:
  - pip install git+https://github.com/Mininglamp-AI/mano-asr.git
    (provides mlx-audio, transformers, etc. — but NOT the core/ package)

The core/ package lives only in the git repo, not in the pip wheel. So
load() clones the repo (shallow, one-time) to get core/auto_model.py.
"""

import os
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/Mininglamp-AI/mano-asr.git"


def _ensure_repo():
    """Clone the mano-asr repo (shallow) if not already present.

    Returns the path to the repo root (contains core/).
    """
    repo_dir = Path(os.path.expanduser("~/.voiceink/models/mano-asr"))
    if repo_dir.exists() and (repo_dir / "core" / "auto_model.py").exists():
        return str(repo_dir)

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
        check=True,
        capture_output=True,
    )
    return str(repo_dir)


def load(model_dir, vad_model_dir=""):
    """Load AutoModel from the mano-asr repo and model weights.

    Args:
        model_dir: Path to the downloaded Mano-ASR model weights
                   (from HuggingFace — downloaded by the YAML spec's
                   hf_repos download step).
        vad_model_dir: Optional path to FSMN-VAD model. Empty = no VAD.

    Returns:
        An AutoModel instance. Call .generate(audio_path, language=...)
        to transcribe.
    """
    repo_root = _ensure_repo()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from core.auto_model import AutoModel

    vad = vad_model_dir if vad_model_dir else None
    return AutoModel(
        model=model_dir,
        model_type="auto",
        vad_model=vad,
    )
