from __future__ import annotations
import gc
import numpy as np
import pandas as pd

# ModernBERT UTILITY regressor: mean-pooled text representation (optionally concatenated with
# LLM difficulty features) -> 11 utility predictions, trained with MSE on the exact target
# Y_i = 0.85*perf_i - 0.15*cost_i/denom. Regressing utility directly (like the validated old
# system) avoids the winner's-curse that sank the perf-then-argmax router. torch/transformers
# are imported lazily so the module is importable on a CPU-only box. Per-fold checkpointed.


def cache_tag(cfg, n_train: int) -> str:
    has_feats = "f" if getattr(cfg, "_encoder_uses_feats", False) else ""
    return f"util{has_feats}_{n_train}_L{cfg.encoder_max_len}_e{cfg.encoder_epochs}"


def _headtail_trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    return text[:head] + "\n[...]\n" + text[-(max_chars - head):]


def _train_predict(cfg, train_texts, Y, eval_texts, feats_tr=None, feats_eval=None):
    """Fine-tune one ModernBERT utility regressor; return (eval x 11) utility predictions."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.encoder_id)
    max_chars = cfg.encoder_max_len * 6
    n_feat = 0 if feats_tr is None else int(feats_tr.shape[1])

    class DS(Dataset):
        def __init__(self, texts, Y=None, feats=None):
            self.enc = tok([_headtail_trim(t, max_chars) for t in texts],
                           truncation=True, max_length=cfg.encoder_max_len)
            self.Y = Y
            self.feats = feats

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            d = {k: self.enc[k][i] for k in self.enc}
            if self.Y is not None:
                d["labels"] = self.Y[i]
            if self.feats is not None:
                d["feats"] = self.feats[i]
            return d

    def collate(batch):
        has_y = "labels" in batch[0]
        has_f = "feats" in batch[0]
        base = [{k: b[k] for k in b if k not in ("labels", "feats")} for b in batch]
        out = tok.pad(base, return_tensors="pt")
        if has_y:
            out["labels"] = torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32)
        if has_f:
            out["feats"] = torch.tensor(np.stack([b["feats"] for b in batch]), dtype=torch.float32)
        return out

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(cfg.encoder_id)
            h = self.backbone.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(h + n_feat, 11)

        def forward(self, input_ids, attention_mask, feats=None, **kw):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            m = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
            pooled = self.dropout(pooled)
            if feats is not None:
                pooled = torch.cat([pooled, feats.to(pooled.dtype)], dim=1)
            return self.head(pooled)

    model = Router().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.encoder_lr, weight_decay=0.01)
    mse = torch.nn.MSELoss()
    dl = DataLoader(DS(train_texts, Y, feats_tr), batch_size=cfg.encoder_bs,
                    shuffle=True, collate_fn=collate)
    steps = max(1, len(dl) // cfg.encoder_grad_accum) * cfg.encoder_epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device == "cuda"

    model.train()
    opt.zero_grad()
    for epoch in range(cfg.encoder_epochs):
        last = 0.0
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(batch["input_ids"], batch["attention_mask"], batch.get("feats"))
                loss = mse(pred, batch["labels"])
            (loss / cfg.encoder_grad_accum).backward()
            if (step + 1) % cfg.encoder_grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
            last = float(loss.item())
        print(f"    epoch {epoch + 1}/{cfg.encoder_epochs} mse={last:.5f}", flush=True)

    model.eval()
    edl = DataLoader(DS(eval_texts, None, feats_eval), batch_size=cfg.encoder_bs * 2,
                     shuffle=False, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for batch in edl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                p = model(batch["input_ids"], batch["attention_mask"], batch.get("feats"))
            preds.append(p.float().cpu().numpy())
    out = np.vstack(preds).astype(np.float32)
    del model, opt, sched
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def encoder_oof_and_test(cfg, train_df, test_df, Y, folds, feats_tr=None, feats_te=None):
    """5-fold OOF + full-fit test utility predictions, checkpointed per fold. Y is the (N,11)
    utility target; feats_* are optional per-row LLM difficulty features (already scaled)."""
    cfg._encoder_uses_feats = feats_tr is not None
    texts = train_df["query"].fillna("").astype(str).tolist()
    te_texts = test_df["query"].fillna("").astype(str).tolist()
    etag = cache_tag(cfg, len(train_df))
    oof = np.zeros_like(Y, dtype=np.float32)
    for i, (tr, va) in enumerate(folds, 1):
        fp = cfg.cache_dir / f"enc_fold{i}_{etag}.npz"
        if fp.exists():
            z = np.load(fp)
            oof[z["va"]] = z["pred"]
            print(f"[encoder] fold {i}/{len(folds)} loaded from cache", flush=True)
            continue
        print(f"[encoder] fold {i}/{len(folds)} ({len(tr)} train, {len(va)} val)...", flush=True)
        ftr = None if feats_tr is None else feats_tr[tr]
        fva = None if feats_tr is None else feats_tr[va]
        pred = _train_predict(cfg, [texts[j] for j in tr], Y[tr], [texts[j] for j in va], ftr, fva)
        oof[va] = pred
        if not cfg.smoke:
            np.savez(fp, va=va, pred=pred)
    tfp = cfg.cache_dir / f"enc_testfull_{etag}.npy"
    if tfp.exists():
        print("[encoder] test full-fit loaded from cache", flush=True)
        test_pred = np.load(tfp)
    else:
        print("[encoder] full-fit for test predictions...", flush=True)
        test_pred = _train_predict(cfg, texts, Y, te_texts, feats_tr, feats_te)
        if not cfg.smoke:
            np.save(tfp, test_pred)
    return oof, test_pred
