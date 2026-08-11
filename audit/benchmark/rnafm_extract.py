"""Offline GPU extraction of frozen RNA-FM pooled embeddings for the audit
dataset's unique junction sequences.

Writes an .npz cache {seqs: [..], vecs: [..]} consumed by
audit.benchmark.rnafm_features.load_cache.  The encoder is frozen and
unsupervised, so this is label-free and introduces no fold leakage.

Usage (server, GPU):
    python audit/benchmark/rnafm_extract.py \
        /mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl \
        /mnt/cunyuliu/rna_junction_repair_20260811T090000Z/RNA-FM_pretrained.pth \
        /mnt/cunyuliu/rna_junction_repair_20260811T090000Z/rnafm_junction_embeddings.npz \
        --device cuda:6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.rnafm_features import pooled_embedding, RENDER_DIM


def _unique_seqs(canonical_source: Path) -> list[str]:
    seen = set()
    out = []
    for line in canonical_source.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        seq = str(o["junction_seq"])
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out


def main():
    canonical_source, model_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cuda:0"
    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device") + 1]

    import torch
    import fm

    model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    model.eval()
    model.to(device)
    print(f"loaded RNA-FM layers={model.args.layers} hidden={model.args.embed_dim} device={device}",
          flush=True)

    seqs = _unique_seqs(Path(canonical_source))
    print(f"unique junction sequences: {len(seqs)}", flush=True)
    vecs = np.zeros((len(seqs), RENDER_DIM), dtype=np.float64)
    conv = alphabet.get_batch_converter()

    B = 64
    for start in range(0, len(seqs), B):
        chunk = seqs[start:start + B]
        # The junction sequences are two RNA arms joined by '_' (e.g. "CUAG_CUAAG").
        # '_' is not in the RNA-FM alphabet (it would map to `<unk>`), so encode it
        # as the valid trained gap token '-' to give the junction boundary a distinct
        # embedding.  The cache is still keyed by the ORIGINAL sequence (with '_').
        batch = [(str(i), s.replace("_", "-")) for i, s in enumerate(chunk)]
        _, _, tokens = conv(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            rep = model(tokens, repr_layers=[model.args.layers])["representations"][model.args.layers]
        pad = alphabet.padding_idx
        m = (tokens != pad).unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1)
        meanp = (rep * m).sum(1) / denom
        maxp = (rep * m + (1 - m) * -1e9).max(1).values
        cls = rep[:, 0, :]
        cat = torch.cat([meanp, maxp, cls], dim=-1).float().cpu().numpy()
        vecs[start:start + B] = cat
        print(f"  extracted {min(start+B, len(seqs))}/{len(seqs)}", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, seqs=np.asarray(seqs, dtype=object), vecs=vecs)
    print(f"wrote cache to {out_path} shape={vecs.shape}", flush=True)


if __name__ == "__main__":
    main()