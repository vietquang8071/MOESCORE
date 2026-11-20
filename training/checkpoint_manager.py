"""Atomic, gating-only checkpoints with strict provenance and safe loading."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import torch

from model.expert_adapters.registry import profile_contract
from model.moe_gating_core import GatingNetwork
from data.source_identity import validate_source_fingerprints


CHECKPOINT_FORMAT_VERSION = 1
FULL9_CHECKPOINT_FORMAT_VERSION = 2
LEGACY_FULL7_CHECKPOINT_FORMAT_VERSION = 3
LEGACY_FULL7_OVERLAP_CHECKPOINT_FORMAT_VERSION = 4
LEGACY_FULL8_OVERLAP_CHECKPOINT_FORMAT_VERSION = 5
FULL7_CHECKPOINT_FORMAT_VERSION = 6
FULL8_CHECKPOINT_FORMAT_VERSION = 7
FULL7_GATE_ARCHITECTURE = {"input_width": 29959, "hidden_width": 512,
                           "normalization": "LayerNorm", "activation": "ReLU",
                           "dropout": 0.1, "output_width": 7, "output": "Softmax(dim=-1)"}
FULL8_GATE_ARCHITECTURE = {"input_width": 33032, "hidden_width": 512,
                           "normalization": "LayerNorm", "activation": "ReLU",
                           "dropout": 0.1, "output_width": 8, "output": "Softmax(dim=-1)"}
_FULL7_SOURCE_FILES = (
    "model/moe_gating_core.py", "training/moe_trainer.py", "training/moe_evaluator.py",
    "training/target_transform.py", "training/checkpoint_manager.py",
    "loss/loss_function.py", "data/full7_assembler.py", "data/full7_score_contract.py",
    "data/moe_feature_store.py", "model/expert_adapters/registry.py",
    "scripts/train_moe_gating.py", "scripts/evaluate_moe_gating.py",
    "scripts/infer_moe_gating.py", "scripts/audit_full7_scores.py",
    "data/source_identity.py", "training/config.py", "training/set_b_trainer.py",
    "training/set_b_config.py", "scripts/train_set_b_residual.py",
    "scripts/build_full7_feature_store.py",
)
_FULL8_SOURCE_FILES = tuple(dict.fromkeys((
    *_FULL7_SOURCE_FILES,
    "data/full8_assembler.py", "data/full8_contract.py",
    "model/expert_adapters/expert8_m2d_clap_adapter.py",
    "model/expert_adapters/expert8_m2d_clap_runtime.py",
    "model/expert_adapters/m2d_lora_runtime.py", "data/unified_dataset.py",
    "data/moe_expert_cache.py", "scripts/extract_moe_feature_store.py",
    "expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py",
    "expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py",
    "scripts/extract_expert8_features.py", "scripts/build_full8_feature_store.py",
    "training/full8_resume_policy.py",
)))
_LEGACY_FULL7_SOURCE_CODE_SHA256 = {
    "model/moe_gating_core.py": "fbc4038ff4579686144de476f81c22b17d431a083208282aa8168617dd77ee6d",
    "training/moe_trainer.py": "13f1c3f8b809ef2fb57aa7e4dff6228e58bbde65388f6d55802909452fecec6c",
    "training/moe_evaluator.py": "26f2de25aaed5ba1df541b573b3257d4384672f0bbfdb108e5a94c7a0fd5f213",
    "training/target_transform.py": "255fc3ab19383ce4e03ac44317398d178ef51ef8a43daea0f66882cd4d8ba2f7",
    "training/checkpoint_manager.py": "4772e92da8c26a01b2b3942d1b24f127ece867cde549a77c0387dc372b8f5f90",
    "loss/loss_function.py": "1e162ed389dee0801c71c97a249c2c848a9fcde3690015526da21003dac9ff56",
    "data/full7_assembler.py": "e89e6da0955e895895942f8e1d468397cbdabd8fb123b826c1d249d1db4700fa",
    "data/full7_score_contract.py": "554f01d382a62b5247651f3f4edce8a9ab80b7a5025748348a131846b49bfba2",
    "data/moe_feature_store.py": "b8eee3bc903a3a4f283fe30784327c359f9a22f5e5c4a4c7b8aa64e7c58da6c7",
    "model/expert_adapters/registry.py": "d1c4f350b9669dd4ebc88b6dafc208a9e85f815a7f06a9b1f9f159252f1a0def",
    "scripts/train_moe_gating.py": "4af42fbad55e33b10feaac54b4425317ecd9c21832e459cf89d2c73ef6223ab2",
    "scripts/evaluate_moe_gating.py": "2567cf1c09405bd07b2a1f77b99bda7206a63feedd71e2cab1532c46bb57f594",
    "scripts/infer_moe_gating.py": "a57d0e5b46fda6c7f76a5ab565080e6e01b6b533ef32d5ba892809a8402edfa4",
    "scripts/audit_full7_scores.py": "7e188f1ea90dfaade180d545d688d8865732489392d5f00c526f308c1b98a53a",
}
# Exact executable source identity at the branch base immediately before Full8.
# Historical format-4 Full7 checkpoints remain usable only when their complete
# recorded map matches this allowlisted source revision or the current code.
_FULL7_BRANCH_BASE_SOURCE_CODE_SHA256 = {
    "model/moe_gating_core.py": "fbc4038ff4579686144de476f81c22b17d431a083208282aa8168617dd77ee6d",
    "training/moe_trainer.py": "13f1c3f8b809ef2fb57aa7e4dff6228e58bbde65388f6d55802909452fecec6c",
    "training/moe_evaluator.py": "26f2de25aaed5ba1df541b573b3257d4384672f0bbfdb108e5a94c7a0fd5f213",
    "training/target_transform.py": "255fc3ab19383ce4e03ac44317398d178ef51ef8a43daea0f66882cd4d8ba2f7",
    "training/checkpoint_manager.py": "e09054cf028647265a01f81c0e0582037f6df36b964ab1dd86260a9f3f75f538",
    "loss/loss_function.py": "1e162ed389dee0801c71c97a249c2c848a9fcde3690015526da21003dac9ff56",
    "data/full7_assembler.py": "e89e6da0955e895895942f8e1d468397cbdabd8fb123b826c1d249d1db4700fa",
    "data/full7_score_contract.py": "554f01d382a62b5247651f3f4edce8a9ab80b7a5025748348a131846b49bfba2",
    "data/moe_feature_store.py": "b8eee3bc903a3a4f283fe30784327c359f9a22f5e5c4a4c7b8aa64e7c58da6c7",
    "model/expert_adapters/registry.py": "d1c4f350b9669dd4ebc88b6dafc208a9e85f815a7f06a9b1f9f159252f1a0def",
    "scripts/train_moe_gating.py": "4b685438b7576f899b5530a4db23e56a68d2cc8cb972c65c1a9dace9d69b0ec1",
    "scripts/evaluate_moe_gating.py": "19a552f0a7b7c1158a7ec308d53f4d33192cb4ab55f3277f3ced20a7033cbe08",
    "scripts/infer_moe_gating.py": "a57d0e5b46fda6c7f76a5ab565080e6e01b6b533ef32d5ba892809a8402edfa4",
    "scripts/audit_full7_scores.py": "7e188f1ea90dfaade180d545d688d8865732489392d5f00c526f308c1b98a53a",
    "data/audio_overlap.py": "207a560b11826813ce4336d14c881224f72cc990de977514b010bde35027069f",
    "training/config.py": "2bf2b3572dd43bb436198d17087419b4bac995778e9771d7d93599b79505c05a",
    "training/set_b_trainer.py": "f1cdd5c5d7cc488fa0ea7e6d8987981a32d638bdd418b32dae6b1dc38ac5d48f",
    "training/set_b_config.py": "090793544788aeed59f99f665ca6b04e27a5e24a324bbfc4b838cf4de941e9fe",
    "scripts/train_set_b_residual.py": "dfed51d0166f68342e9d9cf4d53c3718e19cb531b62cbf3cca440c0890b2b4fe",
    "scripts/build_full7_feature_store.py": "1b74c89256ec93e77258faef7d628c8ba0e0b73b6d9f63058f4fc77309dafe24",
}

# Historical Full7/Full8 source identities from checkpoint formats 4 and 5.
# These allow authenticated inference of older checkpoints only. Their old
# overlap fields are treated as inert historical metadata and never recomputed.
_FULL7_PRE_OVERLAP_SOURCE_CODE_SHA256 = {
    "model/moe_gating_core.py": "fbc4038ff4579686144de476f81c22b17d431a083208282aa8168617dd77ee6d",
    "training/moe_trainer.py": "13f1c3f8b809ef2fb57aa7e4dff6228e58bbde65388f6d55802909452fecec6c",
    "training/moe_evaluator.py": "26f2de25aaed5ba1df541b573b3257d4384672f0bbfdb108e5a94c7a0fd5f213",
    "training/target_transform.py": "255fc3ab19383ce4e03ac44317398d178ef51ef8a43daea0f66882cd4d8ba2f7",
    "training/checkpoint_manager.py": "8d5dfbc697e980348d2ea61d497fa95f90203ccf533bc59db27a913b8855542e",
    "loss/loss_function.py": "1e162ed389dee0801c71c97a249c2c848a9fcde3690015526da21003dac9ff56",
    "data/full7_assembler.py": "e89e6da0955e895895942f8e1d468397cbdabd8fb123b826c1d249d1db4700fa",
    "data/full7_score_contract.py": "554f01d382a62b5247651f3f4edce8a9ab80b7a5025748348a131846b49bfba2",
    "data/moe_feature_store.py": "9ecf480fcc46db36e7099af804f1154c67de1e2dbbfd9d53e7d4e5f05f830399",
    "model/expert_adapters/registry.py": "87533bbe92165145d79bf015d500e7cca5e983fac2ddcb6c39ed7414bba92c95",
    "scripts/train_moe_gating.py": "67b663c23e514097218dd0367ec02b2c0f4fdf3871b576696d691be73223a6d3",
    "scripts/evaluate_moe_gating.py": "5b4d1eb7c1dbff5c78be43e640185600fc957814cd1d5f943624c9910b770f32",
    "scripts/infer_moe_gating.py": "d346aacb7e8cd63c73b1bf294148e781f367fd014bcbe712a157151559e87c33",
    "scripts/audit_full7_scores.py": "1839d1673fea22d3db0f2e2f06cb9a2ea1b6a7903ca9b0b12a306c710a0d5bef",
    "data/audio_overlap.py": "207a560b11826813ce4336d14c881224f72cc990de977514b010bde35027069f",
    "training/config.py": "a70cfbf4332af2aa566526dad1ff5fd16811604f6a9db4581efa4fb8ca74a5b3",
    "training/set_b_trainer.py": "f1cdd5c5d7cc488fa0ea7e6d8987981a32d638bdd418b32dae6b1dc38ac5d48f",
    "training/set_b_config.py": "090793544788aeed59f99f665ca6b04e27a5e24a324bbfc4b838cf4de941e9fe",
    "scripts/train_set_b_residual.py": "dfed51d0166f68342e9d9cf4d53c3718e19cb531b62cbf3cca440c0890b2b4fe",
    "scripts/build_full7_feature_store.py": "1b74c89256ec93e77258faef7d628c8ba0e0b73b6d9f63058f4fc77309dafe24",
}
_FULL8_PRE_OVERLAP_SOURCE_CODE_SHA256 = {
    **_FULL7_PRE_OVERLAP_SOURCE_CODE_SHA256,
    "data/full8_assembler.py": "627b46635c33aba03540b77cad6827429e687c7c4419d50b58b0bb8cbff05da4",
    "data/full8_contract.py": "9d94d5c39a848d688196e98ccdf563982f7fa1c54d95b3a54191e9073c8cb78d",
    "model/expert_adapters/expert8_m2d_clap_adapter.py": "27604bd2297bf420bff778251da066263e75d619d66de2508ba1848de019f702",
    "model/expert_adapters/expert8_m2d_clap_runtime.py": "1ef0172a2466e53373d375c9d93404efaa07ab3fbc6045983b974d854cfdec39",
    "model/expert_adapters/m2d_lora_runtime.py": "d8f971344df56e1052c88f3a333155f5176b35576adb83f6391b6286a44bbebc",
    "data/unified_dataset.py": "ecfe54128a046a798ff3ad9b7674e29badb6c441443ac7dd77e757c301cedabf",
    "data/moe_expert_cache.py": "a9fe38389d736dd1849945e95bca0af70645e47bfb1a4efab0c3751a19b66153",
    "scripts/extract_moe_feature_store.py": "adfc5b988d292c4fbc956d98e3b350e5b6c05ea06a2b95ef0bdbb59159324b83",
    "expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py": "d6a715e5c0fd17ebb348ed76c71c1122333910416a980bb0f6b886c01a1fac21",
    "expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py": "cf300bf1b875b0e19968a916314659e5d66db131054a25d91480ca632abdc89a",
    "scripts/extract_expert8_features.py": "88e91eaab015ba99f967ec0a9f403be6c735b31ef98131c0a29753ed7a86a70b",
    "scripts/build_full8_feature_store.py": "7641367599ac8b85a8fb106baa1015f0fe5cab0b3ffcd7050c14d3e5f932633d",
    "training/full8_resume_policy.py": "321e0d05fb858bb29f6095d79e71b6d8caaf7fc6359944cb70566f31c05d15ba",
}


def full7_source_code_sha256() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in _FULL7_SOURCE_FILES}


def full8_source_code_sha256() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in _FULL8_SOURCE_FILES}


def _validate_gate_payload(payload: Mapping[str, Any], gate: GatingNetwork) -> None:
    version = payload.get("format_version")
    if version not in {CHECKPOINT_FORMAT_VERSION, FULL9_CHECKPOINT_FORMAT_VERSION,
                       LEGACY_FULL7_CHECKPOINT_FORMAT_VERSION,
                       LEGACY_FULL7_OVERLAP_CHECKPOINT_FORMAT_VERSION,
                       LEGACY_FULL8_OVERLAP_CHECKPOINT_FORMAT_VERSION,
                       FULL7_CHECKPOINT_FORMAT_VERSION, FULL8_CHECKPOINT_FORMAT_VERSION}:
        raise ValueError("unsupported gating checkpoint schema")
    profile = "top1_4" if version == CHECKPOINT_FORMAT_VERSION else payload.get("expert_profile")
    order, dimensions, width = profile_contract(profile)
    if profile != gate.profile or payload.get("gate_input_dim") != width or payload.get("gate_expert_count") != len(order):
        raise ValueError("checkpoint expert profile, feature width, or expert count is incompatible")
    if version == FULL9_CHECKPOINT_FORMAT_VERSION and (
        payload.get("expert_order") != list(order)
        or payload.get("expert_feature_dimensions") != dimensions
        or not isinstance(payload.get("expert_checkpoint_provenance"), Mapping)
        or set(payload["expert_checkpoint_provenance"]) != set(order)
        or not isinstance(payload.get("feature_store_identity"), Mapping)
        or not all(payload["feature_store_identity"].get(key) for key in ("train", "validation", "preprocessing_contract"))
    ):
        raise ValueError("checkpoint full_9 order, dimensions, or provenance are incompatible")
    if version in {LEGACY_FULL7_CHECKPOINT_FORMAT_VERSION,
                   LEGACY_FULL7_OVERLAP_CHECKPOINT_FORMAT_VERSION,
                   FULL7_CHECKPOINT_FORMAT_VERSION} and (
        profile != "full_7" or payload.get("expert_order") != list(order)
        or payload.get("expert_feature_dimensions") != dimensions
        or payload.get("gate_architecture") != FULL7_GATE_ARCHITECTURE
        or payload.get("selection_policy") != "official_validation_srcc_strict_v1_provisional"
        or not isinstance(payload.get("expert_checkpoint_provenance"), Mapping)
        or set(payload["expert_checkpoint_provenance"]) != set(order)
        or not isinstance(payload.get("feature_store_identity"), Mapping)
        or not all(payload["feature_store_identity"].get(key) for key in
                   ("train", "validation", "preprocessing_contract"))
        or (version == LEGACY_FULL7_CHECKPOINT_FORMAT_VERSION
            and not _source_hashes_are_valid(payload["feature_store_identity"]))
        or (version in {LEGACY_FULL7_OVERLAP_CHECKPOINT_FORMAT_VERSION,
                        FULL7_CHECKPOINT_FORMAT_VERSION}
            and not _source_membership_is_valid(payload["feature_store_identity"]))
    ):
        raise ValueError("checkpoint full_7 architecture, source code, order, or provenance are incompatible")
    if version in {LEGACY_FULL8_OVERLAP_CHECKPOINT_FORMAT_VERSION, FULL8_CHECKPOINT_FORMAT_VERSION} and (
        profile != "full_8" or payload.get("expert_order") != list(order)
        or payload.get("expert_feature_dimensions") != dimensions
        or payload.get("gate_architecture") != FULL8_GATE_ARCHITECTURE
        or payload.get("selection_policy") != "official_validation_srcc_strict_v1_provisional"
        or not isinstance(payload.get("expert_checkpoint_provenance"), Mapping)
        or set(payload["expert_checkpoint_provenance"]) != set(order)
        or not isinstance(payload.get("feature_store_identity"), Mapping)
        or not all(payload["feature_store_identity"].get(key) for key in
                   ("train", "validation", "preprocessing_contract"))
        or not _source_membership_is_valid(payload["feature_store_identity"])
    ):
        raise ValueError("checkpoint full_8 architecture, source, order, or provenance are incompatible")
    if version == LEGACY_FULL7_CHECKPOINT_FORMAT_VERSION:
        if payload.get("source_code_sha256") != _LEGACY_FULL7_SOURCE_CODE_SHA256:
            raise ValueError("legacy Full7 checkpoint source code identity is incompatible")
    if version == LEGACY_FULL7_OVERLAP_CHECKPOINT_FORMAT_VERSION:
        if payload.get("source_code_sha256") not in (
            _FULL7_PRE_OVERLAP_SOURCE_CODE_SHA256,
            _FULL7_BRANCH_BASE_SOURCE_CODE_SHA256,
        ):
            raise ValueError("checkpoint full_7 source code identity is incompatible")
        identity = payload.get("feature_store_identity")
        config = payload.get("config")
        _validate_full7_review_identity(identity, config)
    if version == FULL7_CHECKPOINT_FORMAT_VERSION:
        if payload.get("source_code_sha256") != full7_source_code_sha256():
            raise ValueError("checkpoint full_7 source code identity is incompatible")
        identity = payload.get("feature_store_identity")
        config = payload.get("config")
        _validate_full7_review_identity(identity, config)
    if version == LEGACY_FULL8_OVERLAP_CHECKPOINT_FORMAT_VERSION:
        if payload.get("source_code_sha256") != _FULL8_PRE_OVERLAP_SOURCE_CODE_SHA256:
            raise ValueError("checkpoint full_8 source code identity is incompatible")
        identity = payload.get("feature_store_identity")
        config = payload.get("config")
        data_config = config.get("data") if isinstance(config, Mapping) else None
        if not isinstance(data_config, Mapping) or data_config.get("profile") != "full_8":
            raise ValueError("checkpoint Full8 profile configuration is invalid")
        _validate_full7_review_identity(identity, config)
    if version == FULL8_CHECKPOINT_FORMAT_VERSION:
        if payload.get("source_code_sha256") != full8_source_code_sha256():
            raise ValueError("checkpoint full_8 source code identity is incompatible")
        identity = payload.get("feature_store_identity")
        config = payload.get("config")
        data_config = config.get("data") if isinstance(config, Mapping) else None
        if not isinstance(data_config, Mapping) or data_config.get("profile") != "full_8":
            raise ValueError("new Full8 checkpoint configuration is invalid")
        _validate_full7_review_identity(identity, config)


def _canonical(value: Any, field: str) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must contain only JSON-safe finite values") from exc


def _fingerprint(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field)).hexdigest()


def _source_membership_is_valid(identity: Mapping[str, Any]) -> bool:
    hashes = identity.get("source_wav_sha256_by_split")
    sample_ids = identity.get("source_sample_ids_by_split")
    if (not isinstance(hashes, Mapping) or set(hashes) != {"train", "validation"}
            or not isinstance(sample_ids, Mapping) or set(sample_ids) != {"train", "validation"}):
        return False
    for split in ("train", "validation"):
        split_hashes, split_ids = hashes[split], sample_ids[split]
        if (not isinstance(split_hashes, list) or not isinstance(split_ids, list)
                or len(split_hashes) != len(split_ids)):
            return False
        try:
            rows = [{"sample_id": sample_id, "wav_sha256": digest}
                    for sample_id, digest in zip(split_ids, split_hashes)]
            normalized = validate_source_fingerprints(rows, split=split)
        except (TypeError, ValueError):
            return False
        if len(normalized) != len(split_ids):
            return False
    return True


def _source_hashes_are_valid(identity: Mapping[str, Any]) -> bool:
    hashes = identity.get("source_wav_sha256_by_split")
    if not isinstance(hashes, Mapping) or set(hashes) != {"train", "validation"}:
        return False
    return all(
        isinstance(values, list)
        and all(isinstance(digest, str) and len(digest) == 64
                and all(char in "0123456789abcdefABCDEF" for char in digest)
                for digest in values)
        for values in hashes.values()
    )


def _validate_full7_review_identity(identity: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    policy = identity.get("score_review_policy")
    allowed = {"EXPERIMENTAL_SKIPPED", "HUMAN_REVIEWED", "SYNTHETIC_NOT_APPLICABLE"}
    if not isinstance(config, Mapping) or not isinstance(policy, str) or policy not in allowed:
        raise ValueError("Full7 checkpoint score review policy is missing or invalid")
    data_config = config.get("data")
    if not isinstance(data_config, Mapping) or data_config.get("score_review_policy") != policy:
        raise ValueError("Full7 checkpoint score review policy differs from its run configuration")
    review_sha256 = identity.get("score_review_sha256")
    if policy == "HUMAN_REVIEWED":
        if (not isinstance(review_sha256, str) or len(review_sha256) != 64
                or any(char not in "0123456789abcdef" for char in review_sha256)):
            raise ValueError("HUMAN_REVIEWED Full7 checkpoint requires the verified score-review SHA-256")
    elif review_sha256 is not None:
        raise ValueError(f"{policy} Full7 checkpoint cannot contain a human score-review SHA-256")
    return policy


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class MoeCheckpointManager:
    """Persist gate and training state without serializing expert modules."""

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        gating_network: GatingNetwork,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        global_step: int,
        config: Mapping[str, Any],
        feature_store_identity: Mapping[str, str],
        expert_checkpoint_provenance: Mapping[str, Any],
        sampler_state: Optional[Mapping[str, Any]] = None,
        best_srcc: Optional[float] = None,
        patience: int = 0,
        seed: int = 2027,
    ) -> Path:
        if not isinstance(gating_network, GatingNetwork):
            raise ValueError("checkpoint must contain a verified gating network")
        order, dimensions, width = profile_contract(gating_network.profile)
        if gating_network.input_feature_dim != width or gating_network.num_experts != len(order):
            raise ValueError("gating network does not match its expert profile")
        if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
            raise ValueError("epoch and global_step must be non-negative integers")
        if type(patience) is not int or patience < 0 or type(seed) is not int or seed < 0:
            raise ValueError("patience and seed must be non-negative integers")
        if best_srcc is not None and (not isinstance(best_srcc, (int, float)) or not torch.isfinite(torch.tensor(float(best_srcc))).item()):
            raise ValueError("best_srcc must be finite or None")
        config_body = json.loads(_canonical(dict(config), "config").decode("utf-8"))
        identity_body = json.loads(_canonical(dict(feature_store_identity), "feature_store_identity").decode("utf-8"))
        provenance_body = json.loads(_canonical(dict(expert_checkpoint_provenance), "expert_checkpoint_provenance").decode("utf-8"))
        if not identity_body or not provenance_body:
            raise ValueError("feature-store identity and expert checkpoint provenance are required")
        if gating_network.profile in {"full_9", "full_7", "full_8"} and (
            set(provenance_body) != set(order)
            or not all(identity_body.get(key) for key in ("train", "validation", "preprocessing_contract"))
        ):
            raise ValueError("full profile checkpoint requires complete expert and store provenance")
        if gating_network.profile in {"full_7", "full_8"} and not _source_membership_is_valid(identity_body):
            raise ValueError(f"{gating_network.profile} checkpoint source membership is invalid")
        if gating_network.profile == "full_7":
            _validate_full7_review_identity(identity_body, config_body)
        if gating_network.profile == "full_8":
            data_config = config_body.get("data")
            if not isinstance(data_config, Mapping) or data_config.get("profile") != "full_8":
                raise ValueError("new Full8 checkpoint configuration is invalid")
            _validate_full7_review_identity(identity_body, config_body)
        if sampler_state is not None:
            sampler_body = json.loads(_canonical(dict(sampler_state), "sampler_state").decode("utf-8"))
        else:
            sampler_body = None
        gate_state = {key: value.detach().cpu().clone() for key, value in gating_network.state_dict().items()}
        payload = {
            "format_version": {"top1_4": CHECKPOINT_FORMAT_VERSION,
                               "full_9": FULL9_CHECKPOINT_FORMAT_VERSION,
                               "full_7": FULL7_CHECKPOINT_FORMAT_VERSION,
                               "full_8": FULL8_CHECKPOINT_FORMAT_VERSION}[gating_network.profile],
            "gate_input_dim": gating_network.input_feature_dim,
            "gate_expert_count": gating_network.num_experts,
            "gating_state_dict": gate_state,
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "scheduler_state_dict": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": config_body,
            "config_sha256": _fingerprint(config_body, "config"),
            "feature_store_identity": identity_body,
            "feature_store_identity_sha256": _fingerprint(identity_body, "feature_store_identity"),
            "expert_checkpoint_provenance": provenance_body,
            "expert_checkpoint_provenance_sha256": _fingerprint(provenance_body, "expert_checkpoint_provenance"),
            "sampler_state": sampler_body,
            "best_srcc": float(best_srcc) if best_srcc is not None else None,
            "patience": patience,
            "seed": seed,
            "torch_rng_state": torch.get_rng_state().cpu().clone(),
            "cuda_rng_states": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.is_initialized() else [],
        }
        if gating_network.profile in {"full_9", "full_7", "full_8"}:
            payload.update({
                "expert_profile": gating_network.profile,
                "expert_order": list(order),
                "expert_feature_dimensions": dimensions,
                "training_config": config_body,
                "preprocessing_identity": identity_body.get("preprocessing_contract"),
            })
        if gating_network.profile == "full_7":
            payload.update({"gate_architecture": FULL7_GATE_ARCHITECTURE,
                            "selection_policy": "official_validation_srcc_strict_v1_provisional",
                            "source_code_sha256": full7_source_code_sha256()})
        if gating_network.profile == "full_8":
            payload.update({"gate_architecture": FULL8_GATE_ARCHITECTURE,
                            "selection_policy": "official_validation_srcc_strict_v1_provisional",
                            "source_code_sha256": full8_source_code_sha256()})
        target = Path(path).expanduser().resolve()
        _atomic_torch_save(payload, target)
        return target

    def load(
        self,
        path: str | os.PathLike[str],
        *,
        gating_network: GatingNetwork,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        expected_config: Mapping[str, Any],
        expected_feature_store_identity: Mapping[str, str],
        expected_expert_checkpoint_provenance: Mapping[str, Any],
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"gating checkpoint is missing: {target}")
        try:
            payload = torch.load(target, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"gating checkpoint cannot be safely loaded: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("unsupported gating checkpoint schema")
        _validate_gate_payload(payload, gating_network)
        if payload.get("format_version") in {
            LEGACY_FULL7_CHECKPOINT_FORMAT_VERSION,
            LEGACY_FULL7_OVERLAP_CHECKPOINT_FORMAT_VERSION,
            LEGACY_FULL8_OVERLAP_CHECKPOINT_FORMAT_VERSION,
        }:
            raise ValueError(
                "legacy Full7/Full8 gate checkpoint cannot resume under the current source/store identity schema"
            )
        if payload.get("format_version") == FULL7_CHECKPOINT_FORMAT_VERSION:
            actual_policy = payload["feature_store_identity"].get("score_review_policy")
            expected_policy = expected_feature_store_identity.get("score_review_policy")
            if actual_policy != expected_policy:
                raise ValueError("gating checkpoint score review policy mismatch")
        if payload.get("format_version") == FULL8_CHECKPOINT_FORMAT_VERSION:
            actual_policy = payload["feature_store_identity"].get("score_review_policy")
            expected_policy = expected_feature_store_identity.get("score_review_policy")
            if actual_policy != expected_policy:
                raise ValueError("Full8 gating checkpoint score review policy mismatch")
        expected_values = (
            ("config", dict(expected_config), "configuration"),
            ("feature_store_identity", dict(expected_feature_store_identity), "feature store identity"),
            ("expert_checkpoint_provenance", dict(expected_expert_checkpoint_provenance), "checkpoint provenance"),
        )
        for field, expected, label in expected_values:
            _canonical(expected, label)
            if payload.get(field) != expected:
                raise ValueError(f"gating checkpoint {label} mismatch")
        if payload.get("config_sha256") != _fingerprint(payload["config"], "config"):
            raise ValueError("checkpoint configuration fingerprint mismatch")
        if payload.get("feature_store_identity_sha256") != _fingerprint(payload["feature_store_identity"], "feature_store_identity"):
            raise ValueError("checkpoint feature-store fingerprint mismatch")
        if payload.get("expert_checkpoint_provenance_sha256") != _fingerprint(payload["expert_checkpoint_provenance"], "expert_checkpoint_provenance"):
            raise ValueError("checkpoint expert provenance fingerprint mismatch")
        state_dict = payload.get("gating_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no gating state dictionary")
        try:
            gating_network.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"gating state dictionary is incompatible: {exc}") from exc
        if not isinstance(payload.get("optimizer_state_dict"), dict):
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is None:
            if payload.get("scheduler_state_dict") is not None:
                raise ValueError("checkpoint contains a scheduler but none was configured")
        elif not isinstance(payload.get("scheduler_state_dict"), dict):
            raise ValueError("checkpoint is missing the configured scheduler state")
        else:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if restore_rng:
            rng_state = payload.get("torch_rng_state")
            if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8:
                raise ValueError("checkpoint has invalid PyTorch RNG state")
            torch.set_rng_state(rng_state.cpu())
            cuda_states = payload.get("cuda_rng_states", [])
            if cuda_states and torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.set_rng_state_all(cuda_states)
        return payload

    def load_for_inference(
        self,
        path: str | os.PathLike[str],
        *,
        gating_network: GatingNetwork,
        expected_expert_checkpoint_provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load only gate weights after validating checkpoint/source provenance."""
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"gating checkpoint is missing: {target}")
        try:
            payload = torch.load(target, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"gating checkpoint cannot be safely loaded: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("unsupported gating checkpoint schema")
        _validate_gate_payload(payload, gating_network)
        expected_provenance = dict(expected_expert_checkpoint_provenance)
        _canonical(expected_provenance, "expected_expert_checkpoint_provenance")
        if payload.get("expert_checkpoint_provenance") != expected_provenance:
            raise ValueError("gating checkpoint provenance mismatch with the inference feature store")
        if payload.get("expert_checkpoint_provenance_sha256") != _fingerprint(payload["expert_checkpoint_provenance"], "expert_checkpoint_provenance"):
            raise ValueError("checkpoint expert provenance fingerprint mismatch")
        if payload.get("config_sha256") != _fingerprint(payload.get("config"), "config"):
            raise ValueError("checkpoint configuration fingerprint mismatch")
        if payload.get("feature_store_identity_sha256") != _fingerprint(payload.get("feature_store_identity"), "feature_store_identity"):
            raise ValueError("checkpoint training feature-store fingerprint mismatch")
        state_dict = payload.get("gating_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no gating state dictionary")
        try:
            gating_network.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"gating state dictionary is incompatible: {exc}") from exc
        return payload


__all__ = ["CHECKPOINT_FORMAT_VERSION", "FULL7_CHECKPOINT_FORMAT_VERSION",
           "FULL8_CHECKPOINT_FORMAT_VERSION", "MoeCheckpointManager",
           "full7_source_code_sha256", "full8_source_code_sha256"]
