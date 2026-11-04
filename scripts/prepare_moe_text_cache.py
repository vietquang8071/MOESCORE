#!/usr/bin/env python3
"""Prepare exact model IDs used by the original experts before offline inference.

Run this as a fresh process: Hub/Transformers environment constants may already
have been imported with different offline/cache settings in a notebook kernel.
No pretrained encoder is instantiated, and no expert checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# E2 uses the short ID literally in its native YAML; E3 uses the namespaced ID.
# They must each be resolvable offline under the name the original source uses.
TEXT_CACHE_RESOURCES = (
    ("roberta-base", True),
    ("bert-base-uncased", False),
    ("google-bert/bert-base-uncased", True),
)


def prepare_text_cache(cache_dir, *, allow_download=False):
    cache = str(Path(cache_dir).resolve())
    offline = "0" if allow_download else "1"
    os.environ.update(HF_HOME=cache, HF_HUB_CACHE=cache, TRANSFORMERS_CACHE=cache,
                      HF_HUB_OFFLINE=offline, TRANSFORMERS_OFFLINE=offline, HF_DATASETS_OFFLINE="1")
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from transformers import AutoConfig, AutoTokenizer, BertConfig, BertTokenizer

    checked = []
    for model_id, needs_weights in TEXT_CACHE_RESOURCES:
        try:
            if allow_download:
                patterns = ["*.json", "*.txt", "*.model"]
                if needs_weights:
                    patterns += ["*.safetensors", "pytorch_model.bin"]
                snapshot_download(model_id, cache_dir=cache, allow_patterns=patterns)
            if model_id == "bert-base-uncased":
                # Match E2's original calls, including default cache lookup:
                # BertConfig / BertTokenizer, no explicit cache_dir argument.
                model_config = BertConfig.from_pretrained(model_id, local_files_only=True)
                tokenizer = BertTokenizer.from_pretrained(model_id, local_files_only=True)
            else:
                model_config = AutoConfig.from_pretrained(model_id, cache_dir=cache, local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, local_files_only=True)
            if needs_weights and not any(isinstance(try_to_load_from_cache(model_id, file, cache_dir=cache), str)
                                         for file in ("model.safetensors", "pytorch_model.bin")):
                raise FileNotFoundError(f"Missing pretrained text model weights for {model_id}")
            checked.append({"model_id": model_id, "model_type": model_config.model_type,
                            "tokenizer_class": type(tokenizer).__name__, "weights_required": needs_weights})
        except Exception as exc:
            raise RuntimeError(f"Missing/unreadable approved text cache {model_id} in {cache}. "
                               "Prepare this exact model ID with Internet enabled, or attach its complete cache. "
                               "Keep the original expert configuration unchanged.") from exc
    return {"status": "PASS", "cache_dir": cache, "model_ids": [item["model_id"] for item in checked],
            "resources": checked, "offline_extraction": True, "model_construction": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)
    report = prepare_text_cache(args.cache_dir, allow_download=args.download)
    if args.report_json:
        target = Path(args.report_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
