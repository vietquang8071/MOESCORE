"""Official metadata joined to immutable Full8 cache rows by split-qualified ID."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.moe_feature_store import MoeFeatureStoreDataset
from data.moe_training_dataset import collate_moe_features


DOMAIN_NAMES = (
    "audiocaps", "audiogen", "audioldm", "audioldm2", "make-an-audio2",
    "stable_audio", "tango", "tango2", "tangoflux",
)
EXPECTED_COUNTS = {
    "train": {0: 2500, 1: 750, 2: 500, 3: 500, 4: 750, 5: 750, 6: 500, 7: 500, 8: 750},
    "validation": {0: 1000, **{i: 250 for i in range(1, 9)}},
    "test": {0: 1000, **{i: 250 for i in range(1, 9)}},
}


def load_aligned_metadata(path, manifest, split, *, expected_counts=None):
    """Return rows in cache order, rejecting missing, duplicate or mismatched pairs."""
    expected_counts = EXPECTED_COUNTS[split] if expected_counts is None else expected_counts
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {name.lower(): name for name in (reader.fieldnames or [])}
        name_field = fields.get("wav_file_name") or fields.get("wave_file_name")
        id_field = fields.get("model_id")
        score_field = next((fields[key] for key in (
            "average_score", "average_semantic_alignment_score", "average_semantic_alignement_score"
        ) if key in fields), None)
        text_field = fields.get("text") or fields.get("caption")
        model_name_field = fields.get("model_name")
        if not all((name_field, id_field, score_field, text_field, model_name_field)):
            raise ValueError(f"{split}: metadata needs WAV, model ID/name, text and average score columns")
        by_id = {}
        for row in reader:
            filename = row[name_field].strip()
            sample_id = f"{split}:{filename}"
            if not filename or Path(filename).name != filename or sample_id in by_id:
                raise ValueError(f"{split}: duplicate or empty sample identity {sample_id!r}")
            try:
                model_id = int(row[id_field])
                label = float(row[score_field])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{sample_id}: invalid model ID or raw score") from exc
            if model_id not in expected_counts or row[model_name_field].strip() != DOMAIN_NAMES[model_id]:
                raise ValueError(f"{sample_id}: domain ID/name mapping mismatch")
            if not 0 <= label <= 10 or not torch.isfinite(torch.tensor(label)):
                raise ValueError(f"{sample_id}: raw label is outside finite [0,10]")
            by_id[sample_id] = {
                "sample_id": sample_id, "wave_file_name": filename,
                "model_id": model_id, "model_name": DOMAIN_NAMES[model_id],
                "label_raw": label, "caption": row[text_field],
            }
    ids = [sid for shard in manifest["shards"] for sid in shard["sample_ids"]]
    names = [name for shard in manifest["shards"] for name in shard["wave_file_names"]]
    fingerprints = manifest["assembly_identity"]["ordered_source_fingerprints"]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id) or len(ids) != len(names) or len(ids) != len(fingerprints):
        raise ValueError(f"{split}: metadata/cache sample identity alignment differs")
    counts = Counter(row["model_id"] for row in by_id.values())
    if dict(counts) != expected_counts:
        raise ValueError(f"{split}: actual domain counts {dict(counts)} differ from {expected_counts}")
    ordered = []
    for sid, name, fingerprint in zip(ids, names, fingerprints):
        row = by_id[sid]
        if row["wave_file_name"] != name or fingerprint["sample_id"] != sid:
            raise ValueError(f"{sid}: cache filename/source fingerprint differs")
        if hashlib.sha256(row["caption"].encode("utf-8")).hexdigest() != fingerprint["caption_sha256"]:
            raise ValueError(f"{sid}: official caption differs from cached audio/text pairing")
        ordered.append(row)
    return ordered


class DSSDataset(Dataset):
    def __init__(self, store_root, metadata_path, *, split):
        self.store = MoeFeatureStoreDataset(store_root, split=split, require_labels=split != "test")
        if self.store.manifest.get("expert_profile") != "full_8":
            raise ValueError(f"{split}: DSS requires a Full8 feature store")
        self.split = split
        self.manifest = self.store.manifest
        self.metadata = load_aligned_metadata(metadata_path, self.manifest, split)
        if len(self.metadata) != len(self.store):
            raise ValueError(f"{split}: cache and official metadata counts differ")
        self.model_ids = [row["model_id"] for row in self.metadata]
        # Compare every cached label before optimization, not only sampled rows.
        if split != "test":
            offset = 0
            for shard_index, shard in enumerate(self.manifest["shards"]):
                labels = self.store._shard_arrays(shard_index)[0].get("labels_raw.npy")
                if labels is None:
                    raise ValueError(f"{split}: cached labels are missing")
                official = np.asarray([row["label_raw"] for row in self.metadata[offset:offset + shard["count"]]])
                if not np.allclose(labels, official, rtol=0, atol=1e-4):
                    bad = int(np.flatnonzero(np.abs(labels - official) > 1e-4)[0])
                    raise ValueError(f"{self.metadata[offset + bad]['sample_id']}: cached label differs from official metadata")
                offset += shard["count"]

    def __len__(self):
        return len(self.store)

    def __getitem__(self, index):
        item = self.store[index]
        row = self.metadata[index]
        item["label_raw"] = torch.tensor(row["label_raw"], dtype=torch.float32)
        item["model_id"] = row["model_id"]
        item["model_name"] = row["model_name"]
        return item


def collate_dss(samples):
    batch = collate_moe_features(samples)
    batch["model_ids"] = torch.tensor([row["model_id"] for row in samples], dtype=torch.int64)
    batch["model_names"] = [row["model_name"] for row in samples]
    return batch
