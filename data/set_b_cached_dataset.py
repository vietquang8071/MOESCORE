"""Read-only join of verified Full9 LAION and Whisper feature caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.moe_expert_cache import validate_expert_cache_identity
from data.moe_feature_store import MoeFeatureStoreDataset
from data.set_a_cached_dataset import _read_source_manifest, _labels_match_float32
from data.whisper_feature_cache import (
    FEATURE_EXTRACTION_VERSION, WhisperFeatureCacheWriter, _canonical_json,
)
from model.composite_experts.set_a_residual import SET_A_FEATURE_BLOCKS, SET_A_FEATURE_REPRESENTATION
from model.expert_adapters.registry import profile_contract


def _whisper_identity_from_manifest(records, split):
    """Reproduce build_whisper_dataset_identity's row serialization using trusted WAV hashes."""
    digest = hashlib.sha256()
    for record in records:
        if not record["raw_caption"].strip():
            raise ValueError("Whisper source caption must be nonempty")
        row = {"sample_id": record["sample_id"],
               "caption_sha256": hashlib.sha256(record["raw_caption"].encode("utf-8")).hexdigest(),
               "audio_sha256": record["wav_sha256"].lower()}
        digest.update(_canonical_json(row))
        digest.update(b"\n")
    return {"split": split, "sample_count": len(records), "ordered_samples_sha256": digest.hexdigest()}


class SetBCachedDataset(Dataset):
    def __init__(self, full9_root, whisper_root, *, split: str, source_manifest_path=None,
                 source_audio_dir=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError("E6 split must be train, validation, or test")
        self.split = split
        self.full9_root = Path(full9_root).expanduser().resolve()
        self.whisper_root = Path(whisper_root).expanduser().resolve()
        self.store = MoeFeatureStoreDataset(self.full9_root, split=split, require_labels=split != "test")
        manifest = self.store.manifest
        order, dimensions, _ = profile_contract("full_9")
        if manifest.get("expert_profile") != "full_9" or manifest.get("expert_order") != list(order) or manifest.get("feature_dimensions") != dimensions:
            raise ValueError("E6 requires current verified full_9 feature representation")
        if split == "test" and manifest.get("labels_present"):
            raise ValueError("E6 test feature store must not contain labels")
        self.laion_index = order.index("top2_laion")
        self.laion_start = sum(dimensions[name] for name in order[:self.laion_index])
        cache = manifest.get("per_expert_cache_identities", {}).get("top2_laion")
        model = manifest.get("checkpoint_provenance", {}).get("top2_laion")
        if not isinstance(cache, dict) or not isinstance(model, dict) or not model:
            raise ValueError("LAION cache/checkpoint provenance missing")
        validate_expert_cache_identity(cache)
        if cache.get("expert_name") != "top2_laion" or cache.get("checkpoint_provenance") != model:
            raise ValueError("LAION cache/checkpoint provenance mismatch")
        contract = cache.get("score_contract", {})
        if (contract.get("feature_representation") != SET_A_FEATURE_REPRESENTATION
                or contract.get("feature_blocks") != list(SET_A_FEATURE_BLOCKS)
                or contract.get("feature_width_formula") != "4D+1"
                or contract.get("feature_shape") != [None, 2049]):
            raise ValueError("LAION feature representation is not current 4D+1")
        input_contract = cache.get("input_contract", {})
        if (input_contract.get("audio_key") != "expert1_wavs"
                or input_contract.get("audio_shape") != [None, 1, 480000]
                or input_contract.get("sample_rate_hz") != 48000):
            raise ValueError("LAION Full9 waveform contract mismatch")
        self.laion_cache_identity = cache
        self.laion_model_identity = model
        source = Path(source_manifest_path).expanduser().resolve() if source_manifest_path else self.full9_root / split / "source_manifest.json"
        self.records = _read_source_manifest(source, split=split, expected_sha256=manifest["source_manifest_sha256"])
        self.sample_ids = [r["sample_id"] for r in self.records]
        names = [r["wave_file_name"] for r in self.records]
        if self.sample_ids != self.store.sample_ids or self.sample_ids != [s for sh in manifest["shards"] for s in sh["sample_ids"]]:
            raise ValueError("Full9 source sample order mismatch")
        if names != [n for sh in manifest["shards"] for n in sh["wave_file_names"]]:
            raise ValueError("Full9 source WAV names mismatch")
        if source_audio_dir is not None:
            audio_dir = Path(source_audio_dir).expanduser().resolve()
            for record in self.records:
                path = audio_dir / record["wave_file_name"]
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["wav_sha256"]:
                    raise ValueError(f"Full9 source WAV changed: {path}")
        identity = _whisper_identity_from_manifest(self.records, split)
        cache_dir = self.whisper_root / split
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Whisper cache manifest missing: {manifest_path}")
        whisper_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = whisper_manifest.get("source_checkpoint_identity", {})
        preprocessing = whisper_manifest.get("preprocessing_identity", {})
        if checkpoint.get("model_id") != "openai/whisper-large-v2":
            raise ValueError("Whisper cache checkpoint is not openai/whisper-large-v2")
        if (not isinstance(preprocessing.get("extraction_batch_size"), int)
                or preprocessing["extraction_batch_size"] <= 0
                or preprocessing.get("decoder_padding_scope") != "per extraction batch; padded positions participate in mean pooling"
                or preprocessing.get("unified_waveform_preprocessing", {}).get("source_sha256") != hashlib.sha256((Path(__file__).parents[1] / "data/unified_dataset.py").read_bytes()).hexdigest()
                or preprocessing.get("collation_source_sha256") != hashlib.sha256((Path(__file__).parents[1] / "data/unified_collate.py").read_bytes()).hexdigest()):
            raise ValueError("Whisper preprocessing or batch-padding identity mismatch")
        self.whisper_reader = WhisperFeatureCacheWriter(cache_dir, split=split,
            source_checkpoint_identity=checkpoint, preprocessing_identity=preprocessing,
            dataset_identity=identity, feature_extraction_version=FEATURE_EXTRACTION_VERSION)
        whisper = self.whisper_reader.load()
        if whisper["sample_ids"] != self.sample_ids or len(whisper["sample_ids"]) != identity["sample_count"]:
            raise ValueError("Whisper cache incomplete or sample order mismatch")
        self.whisper_audio = whisper["whisper_audio_features"]
        self.whisper_text = whisper["whisper_text_features"]
        self.whisper_checkpoint_identity = checkpoint
        self.whisper_preprocessing_identity = preprocessing
        self.provenance = {
            "split": split, "full9_store_identity": self.store.identity,
            "full9_source_manifest_sha256": manifest["source_manifest_sha256"],
            "laion_cache_identity": cache, "laion_model_identity": model,
            "whisper_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "whisper_dataset_identity": identity, "whisper_checkpoint_identity": checkpoint,
            "whisper_preprocessing_identity": preprocessing,
        }

    def __len__(self):
        return len(self.store)

    def __getitem__(self, index):
        row = self.store[index]
        record = self.records[index]
        if row["sample_id"] != record["sample_id"] or row["wave_file_name"] != record["wave_file_name"]:
            raise ValueError("Full9 row identity changed")
        score = row["scores"][self.laion_index:self.laion_index + 1]
        features = row["features"][self.laion_start:self.laion_start + 2049]
        result = {"sample_id": row["sample_id"], "wave_file_name": row["wave_file_name"],
                  "component_outputs": {"top2_laion": {"score": score, "features": features}},
                  "whisper_audio": torch.from_numpy(np.array(self.whisper_audio[index], copy=True)),
                  "whisper_text": torch.from_numpy(np.array(self.whisper_text[index], copy=True))}
        if self.split != "test":
            label = row.get("label_raw")
            if label is None or record.get("label_raw") is None or not _labels_match_float32(label, record["label_raw"], field=f"E6 {self.split} label"):
                raise ValueError("E6 label differs from source manifest")
            result["target_normalized"] = (label - 5.0) / 5.0
        return result
