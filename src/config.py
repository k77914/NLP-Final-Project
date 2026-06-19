from __future__ import annotations
import os, random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _resolve_root() -> Path:
    drive = Path('/content/drive/MyDrive/NLP Final Project')
    if (drive / 'dataset' / 'train.csv').exists():
        return drive
    return Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    seed: int = 42
    root: Path = field(default_factory=_resolve_root)
    n_splits: int = 5
    smoke: bool = False
    smoke_rows: int = 400

    perf_weight: float = 0.85
    cost_weight: float = 0.15

    # features
    tfidf_word_features: int = 100_000
    tfidf_char_features: int = 100_000
    tfidf_max_chars: int = 60_000
    svd_components: int = 256
    embedding_cache_name: str = 'qwen3_4b_dim1024_chunks_v1'
    qwen_model_id: str = 'Qwen/Qwen3-Embedding-4B'
    embedding_dim: int = 1024

    # lgbm
    lgbm_estimators: int = 1200
    lgbm_lr: float = 0.03
    lgbm_leaves: int = 31
    lgbm_min_child: int = 30

    # encoder (Phase 1)
    encoder_id: str = 'answerdotai/ModernBERT-large'
    encoder_max_len: int = 2048
    encoder_epochs: int = 3
    encoder_lr: float = 2e-5
    encoder_bs: int = 8
    encoder_grad_accum: int = 2
    aux_diff_weight: float = 0.3

    # llm difficulty (Phase 2)
    llm_id: str = 'Qwen/Qwen2.5-7B-Instruct'
    llm_k_samples: int = 6
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.8
    hf_batch_size: int = 16  # batch size for the transformers (non-vLLM) generation backend

    @property
    def data_dir(self) -> Path:
        return self.root / 'dataset'

    @property
    def out_dir(self) -> Path:
        d = self.root / 'Output' / 'router_v2'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        d = self.out_dir / 'cache'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def prev_cache_dir(self) -> Path:
        return self.root / 'Output' / 'router_a100' / 'cache'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
