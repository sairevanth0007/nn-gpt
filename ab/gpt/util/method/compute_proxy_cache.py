#!/usr/bin/env python3
"""
Batch-compute zero-cost proxies over the whole preprocessed dataset and cache
the results to disk (Step 2 of IMPROVEMENT_PLAN_Zero_Cost_Proxies.md).

Deduplicates by (nn, dataset) before computing -- proxies are architecture-
intrinsic (they depend on nn_code + input shape, not on hyperparameters or
data transforms), and LEMUR repeats the same architecture across many
(prm_id, transform_id) combinations, so computing per-row would waste most of
the work on exact duplicates (~6350 rows -> ~1550 unique (nn, dataset) pairs).

Resumable by construction: the cache file is JSONL, appended to incrementally,
and already-cached (nn, dataset) keys are skipped on restart.

Two ways to use it:
  * As a library:  build_proxy_cache(input_path, cache_path)   <- called
    automatically by AccPredictor.prepare_llm_datasets when the cache is
    missing, so the whole pipeline runs in one command.
  * Standalone:    python -m ab.gpt.util.method.compute_proxy_cache
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ab.gpt.util.method.zero_cost_proxies import compute_proxies, torch_device, PROXY_NAMES

ACC_DIR = Path(__file__).resolve().parents[4] / "out" / "acc_predict"
INPUT_PATH = ACC_DIR / "llm_finetuning_data.jsonl"
CACHE_PATH = ACC_DIR / "proxy_cache.jsonl"

PROGRESS_EVERY = 100


def _load_unique_records(path: Path) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("nn"), rec.get("dataset"))
            if key in seen or key[0] is None:
                continue
            seen.add(key)
            records.append(rec)
    return records


def _load_cached_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((rec.get("nn"), rec.get("dataset")))
    return keys


def build_proxy_cache(
    input_path: Path = INPUT_PATH,
    cache_path: Path = CACHE_PATH,
    device=None,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """Compute + cache zero-cost proxies for every unique (nn, dataset) in
    input_path, appending to cache_path (resumable). Returns cache_path.

    limit: if set, only compute this many *new* architectures (used for quick
    tests). device: torch device; defaults to torch_device()."""
    input_path = Path(input_path)
    cache_path = Path(cache_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or torch_device()
    if verbose:
        print(f"[proxy-cache] Device: {device}")

    unique_records = _load_unique_records(input_path)
    already_done = _load_cached_keys(cache_path)
    todo = [r for r in unique_records if (r.get("nn"), r.get("dataset")) not in already_done]
    if limit is not None:
        todo = todo[:limit]

    if verbose:
        print(f"[proxy-cache] Unique architectures: {len(unique_records)} | "
              f"already cached: {len(already_done)} | to compute: {len(todo)}")

    if not todo:
        if verbose:
            print("[proxy-cache] Nothing to do -- cache already complete.")
        return cache_path

    ok_count = 0
    fail_reasons: dict[str, int] = {}
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(todo, 1):
            prm = rec.get("prm")
            if isinstance(prm, str):
                try:
                    prm = json.loads(prm)
                except json.JSONDecodeError:
                    prm = {}
            scores = compute_proxies(rec.get("nn_code", ""), rec.get("dataset", ""), prm, device=device)
            row = {"nn": rec.get("nn"), "dataset": rec.get("dataset"), **scores}
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if scores["proxy_status"] == "ok":
                ok_count += 1
            else:
                reason = scores["proxy_status"].split(":")[0].split("(")[0].strip()
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            if verbose and (i % PROGRESS_EVERY == 0 or i == len(todo)):
                elapsed = time.time() - t0
                eta = (len(todo) - i) * (elapsed / i)
                print(f"[proxy-cache] [{i}/{len(todo)}] ok={ok_count} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    if verbose:
        print(f"[proxy-cache] Done. {ok_count}/{len(todo)} newly computed succeeded.")
        if fail_reasons:
            print("[proxy-cache] Failure reasons:", dict(sorted(fail_reasons.items(), key=lambda kv: -kv[1])))
    return cache_path


def main() -> None:
    build_proxy_cache()


if __name__ == "__main__":
    main()
