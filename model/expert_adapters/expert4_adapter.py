"""Source-preserving adapter for the original BEATs/CLAP Expert 4.

The original Expert 4 implementation and its checkpoint loader are imported
only when ``from_finetuned`` is requested. Importing this module does not
initialize a pretrained encoder or read any checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import threading
from types import ModuleType
from typing import Any, Optional, Union

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter


PathValue = Union[str, os.PathLike]
_MAX_AUDIO_SAMPLES = 16_000 * 10
_BEST_CHECKPOINT_KEYS = {
    "model_state_dict",
    "config",
    "fingerprints",
    "epoch",
    "best_srcc",
}
_FINGERPRINT_KEYS = {
    "beats_checkpoint",
    "clap_checkpoint",
    "train_csv",
    "validation_csv",
    "unilm_source",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BEATS_IMPORT_LOCK = threading.RLock()


def _beats_dir(source_dir: PathValue) -> Path:
    source = Path(source_dir).expanduser().resolve(strict=False)
    candidate = source if (source / "BEATs.py").is_file() else source / "beats"
    if not candidate.is_dir() or not (candidate / "BEATs.py").is_file():
        raise FileNotFoundError(
            f"Expert 4 UniLM source must contain BEATs.py at {source}/BEATs.py "
            f"or {source}/beats/BEATs.py; checked {candidate}"
        )
    return candidate.resolve()


def _module_origins(module: Any) -> list[Path]:
    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.append(Path(module_file).resolve(strict=False))
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if spec_origin and spec_origin not in {"built-in", "frozen"}:
        origins.append(Path(spec_origin).resolve(strict=False))
    for package_path in getattr(module, "__path__", ()):
        origins.append(Path(package_path).resolve(strict=False))
    return list(dict.fromkeys(origins))


def validate_expert4_module_origins(
    unilm_source_dir: PathValue,
    *,
    project_root: Optional[PathValue] = None,
    modules: Optional[Mapping[str, Any]] = None,
    require_beats: bool = False,
) -> None:
    """Reject Expert 4 modules already resolved from another source tree.

    The original audio encoder imports a top-level name ``BEATs``. Its vendored
    file uses package-relative imports, so the factory below exposes it under a
    temporary, source-checked alias while constructing the model. This guard
    rejects an existing conflicting global ``BEATs`` binding before that
    compatibility context is entered. It never removes or replaces an
    unrelated module.
    """
    registry = sys.modules if modules is None else modules
    beats_path = _beats_dir(unilm_source_dir)
    expected_vendor_modules = {
        "BEATs": beats_path / "BEATs.py",
        "backbone": beats_path / "backbone.py",
        "modules": beats_path / "modules.py",
    }
    for name, expected_file in expected_vendor_modules.items():
        module = registry.get(name)
        if module is None:
            if name == "BEATs" and require_beats:
                raise ImportError(
                    f"Expert 4 factory did not load top-level BEATs from {expected_file}"
                )
            continue
        origins = _module_origins(module)
        resolved_expected = expected_file.resolve(strict=False)
        if not origins or any(origin != resolved_expected for origin in origins):
            raise ImportError(
                f"Expert 4 namespace {name} resolves outside the configured UniLM source: "
                f"expected {resolved_expected}, found {origins or '<no inspectable origin>'}"
            )

    root = Path(project_root).expanduser().resolve(strict=False) if project_root else _PROJECT_ROOT
    source_root = root / "expert4_finetune"
    expected_modules = {
        "expert4_finetune.model": source_root / "model.py",
        "expert4_finetune.audio_encoder": source_root / "audio_encoder.py",
        "expert4_finetune.text_encoder": source_root / "text_encoder.py",
        "expert4_finetune.seq_coattn": source_root / "seq_coattn.py",
        "expert4_finetune.inference": source_root / "inference.py",
        "expert4_finetune.trainer": source_root / "trainer.py",
    }
    for name, expected_file in expected_modules.items():
        module = registry.get(name)
        if module is None:
            continue
        origins = _module_origins(module)
        if not origins or expected_file.resolve(strict=False) not in origins:
            raise ImportError(
                f"Expert 4 module {name} resolved outside the expected source tree: "
                f"expected {expected_file.resolve(strict=False)}, found {origins or '<unknown>'}"
            )


@contextmanager
def _scoped_beats_import(source_dir: PathValue):
    """Supply source-checked top-level aliases for BEATs and its dependencies.

    ``AudioEncoder`` imports ``BEATs`` by a legacy absolute name. The original
    UniLM revision also uses absolute ``backbone`` and ``modules`` imports,
    while the bundled source uses package-relative imports. Load each exact
    source file in a private package, temporarily bind only the three legacy
    names needed by the original loader, then restore those bindings and any
    source-added ``sys.path`` entries. Existing names from another source are
    rejected before this context changes import state.
    """
    beats_path = _beats_dir(source_dir)
    beats_file = (beats_path / "BEATs.py").resolve()
    with _BEATS_IMPORT_LOCK:
        existing = sys.modules.get("BEATs")
        if existing is not None:
            validate_expert4_module_origins(source_dir)
            package_name = getattr(existing, "__package__", "")
            if not package_name and not all(sys.modules.get(name) is not None for name in ("backbone", "modules")):
                raise ImportError(
                    f"Expert 4 BEATs at {beats_file} has no package context for its "
                    "vendor imports; its original backbone/modules aliases are unavailable"
                )
            previous_path_count = sum(
                1 for entry in sys.path
                if entry and Path(entry).resolve(strict=False) == beats_path
            )
            try:
                yield existing
            finally:
                current_count = sum(
                    1 for entry in sys.path
                    if entry and Path(entry).resolve(strict=False) == beats_path
                )
                while current_count > previous_path_count:
                    for index in range(len(sys.path) - 1, -1, -1):
                        entry = sys.path[index]
                        if entry and Path(entry).resolve(strict=False) == beats_path:
                            sys.path.pop(index)
                            break
                    current_count -= 1
            return

        digest = hashlib.sha256(str(beats_path).encode("utf-8")).hexdigest()[:16]
        package_name = f"_moescore_expert4_beats_{digest}"
        package = sys.modules.get(package_name)
        package_created = package is None
        if package is None:
            package = ModuleType(package_name)
            package.__path__ = [str(beats_path)]
            package.__package__ = package_name
            package.__spec__ = importlib.machinery.ModuleSpec(
                package_name, loader=None, is_package=True
            )
            package.__spec__.submodule_search_locations = [str(beats_path)]
            sys.modules[package_name] = package
        else:
            package_paths = [Path(path).resolve(strict=False) for path in getattr(package, "__path__", ())]
            if package_paths != [beats_path]:
                raise ImportError(
                    f"Private Expert 4 BEATs package {package_name} is already bound "
                    f"to unexpected paths: {package_paths}"
                )

        aliases_added: dict[str, ModuleType] = {}
        private_modules_added: dict[str, ModuleType] = {}
        completed = False
        previous_path_count = sum(
            1 for entry in sys.path
            if entry and Path(entry).resolve(strict=False) == beats_path
        )

        def source_module(name: str) -> Optional[ModuleType]:
            qualified = f"{package_name}.{name}"
            expected = (beats_path / f"{name}.py").resolve()
            if not expected.is_file():
                return None
            top_level = sys.modules.get(name)
            private = sys.modules.get(qualified)
            if private is not None and top_level is not None and private is not top_level:
                raise ImportError(
                    f"Expert 4 namespace {name} is already loaded twice; refusing mixed module objects "
                    f"from {getattr(private, '__file__', None)} and {getattr(top_level, '__file__', None)}"
                )
            if private is None and top_level is not None:
                private = top_level
                sys.modules[qualified] = private
                setattr(package, name, private)
                private_modules_added[qualified] = private
            elif private is None:
                spec = importlib.util.spec_from_file_location(qualified, expected)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create a source loader for Expert 4 {name}: {expected}")
                private = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = private
                setattr(package, name, private)
                private_modules_added[qualified] = private
                try:
                    spec.loader.exec_module(private)
                except Exception as exc:
                    if sys.modules.get(qualified) is private:
                        del sys.modules[qualified]
                    if getattr(package, name, None) is private:
                        delattr(package, name)
                    private_modules_added.pop(qualified, None)
                    raise ImportError(
                        f"Could not import Expert 4 {name} from {beats_path}: {exc}"
                    ) from exc
            origins = _module_origins(private)
            if not origins or any(origin != expected for origin in origins):
                raise ImportError(
                    f"Expert 4 namespace {name} origin mismatch; it resolves outside the configured UniLM source: "
                    f"expected {expected}, found {origins or '<no inspectable origin>'}"
                )
            if top_level is None:
                sys.modules[name] = private
                aliases_added[name] = private
            return private

        try:
            # The original source uses `from modules import ...` inside
            # backbone.py and `from backbone import ...` inside BEATs.py.
            # Load optional dependency files only when present. Some older
            # package-relative fixtures do not need both files; the original
            # UniLM revision provides both and uses these aliases.
            source_module("modules")
            source_module("backbone")
            try:
                module = importlib.import_module(f"{package_name}.BEATs")
            except Exception as exc:
                raise ImportError(
                    f"Could not import the configured BEATs package from {beats_file}: {exc}"
                ) from exc
            if Path(getattr(module, "__file__", "")).resolve(strict=False) != beats_file:
                raise ImportError(
                    f"Expert 4 loaded BEATs from {getattr(module, '__file__', None)}, "
                    f"expected {beats_file}"
                )
            if "BEATs" not in sys.modules:
                sys.modules["BEATs"] = module
                aliases_added["BEATs"] = module
            elif sys.modules["BEATs"] is not module:
                raise ImportError("Expert 4 BEATs alias changed during private package construction")
            validate_expert4_module_origins(source_dir)
            yield module
            completed = True
        finally:
            # Remove only aliases this context added, and only if they still
            # point to our modules. A replacement from another owner is left
            # untouched so an import race fails visibly rather than being erased.
            for name in ("BEATs", "backbone", "modules"):
                alias = aliases_added.get(name)
                if alias is not None and sys.modules.get(name) is alias:
                    del sys.modules[name]
            current_count = sum(
                1 for entry in sys.path
                if entry and Path(entry).resolve(strict=False) == beats_path
            )
            while current_count > previous_path_count:
                for index in range(len(sys.path) - 1, -1, -1):
                    entry = sys.path[index]
                    if entry and Path(entry).resolve(strict=False) == beats_path:
                        sys.path.pop(index)
                        break
                current_count -= 1
            if not completed:
                for name, module in private_modules_added.items():
                    if sys.modules.get(name) is module:
                        del sys.modules[name]
                    child_name = name.rsplit(".", 1)[-1]
                    if getattr(package, child_name, None) is module:
                        delattr(package, child_name)
                if package_created:
                    if sys.modules.get(package_name) is package:
                        del sys.modules[package_name]


def _resolved_resource(
    value: Any,
    *,
    field: str,
    directory: bool = False,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Expert 4 {field} must be an absolute filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Expert 4 {field} must already be resolved to an absolute path; got {path}"
        )
    path = path.resolve(strict=False)
    if directory:
        if not path.is_dir():
            raise NotADirectoryError(f"Expert 4 {field} expected a directory at {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Expert 4 {field} expected a file at {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Expert 4 {field} is not readable: {path}")
    return path


def _target_device(device: Union[str, torch.device]) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Expert 4 requested {target}, but CUDA is unavailable")
        index = target.index if target.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Expert 4 requested {target}, but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
    return target


@contextmanager
def _offline_huggingface(offline: bool):
    if type(offline) is not bool:
        raise TypeError(f"Expert 4 offline must be a boolean; got {offline!r}")
    if not offline:
        raise ValueError(
            "Expert 4 restoration is offline-only; populate local tokenizer/model resources first"
        )
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    old_environment = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = "1"
    module_names = ("huggingface_hub.constants", "transformers.utils.hub")
    old_flags: dict[str, Any] = {}
    for name in module_names:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "HF_HUB_OFFLINE"):
            old_flags[name] = module.HF_HUB_OFFLINE
            module.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        for name, value in old_flags.items():
            module = sys.modules.get(name)
            if module is not None:
                module.HF_HUB_OFFLINE = value
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_best_checkpoint_bundle(state: Any) -> Mapping[str, Any]:
    """Validate the exact inference-best bundle written by Expert4 trainer."""
    if not isinstance(state, Mapping):
        raise TypeError("Expert 4 best checkpoint must be a mapping")
    missing = _BEST_CHECKPOINT_KEYS - set(state)
    unexpected = set(state) - _BEST_CHECKPOINT_KEYS
    if missing or unexpected:
        raise ValueError(
            "Expert 4 checkpoint is not the source best-inference bundle: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}. "
            "Resume checkpoints and raw/pretrained-only states are not accepted."
        )
    trained = state["model_state_dict"]
    if not isinstance(trained, Mapping) or not trained:
        raise ValueError("Expert 4 checkpoint model_state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in trained.items()):
        raise TypeError("Expert 4 model_state_dict must contain only string keys and tensors")
    nonfinite = [key for key, value in trained.items() if not torch.isfinite(value).all().item()]
    if nonfinite:
        raise ValueError(f"Expert 4 fine-tuned tensors contain non-finite values: {nonfinite[:8]}")

    config = state["config"]
    if not isinstance(config, Mapping) or not isinstance(config.get("resources"), Mapping):
        raise ValueError("Expert 4 checkpoint config must contain the source resources mapping")
    fingerprints = state["fingerprints"]
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Expert 4 checkpoint fingerprints must be a mapping")
    missing_fingerprints = _FINGERPRINT_KEYS - set(fingerprints)
    unexpected_fingerprints = set(fingerprints) - _FINGERPRINT_KEYS
    if missing_fingerprints or unexpected_fingerprints:
        raise ValueError(
            "Expert 4 checkpoint fingerprint schema mismatch: "
            f"missing={sorted(missing_fingerprints)}, unexpected={sorted(unexpected_fingerprints)}"
        )
    for key in ("beats_checkpoint", "clap_checkpoint", "train_csv", "validation_csv"):
        if not isinstance(fingerprints[key], str) or not _SHA256_RE.fullmatch(fingerprints[key]):
            raise ValueError(f"Expert 4 fingerprint {key} must be a 64-character SHA-256")
    source_identity = fingerprints["unilm_source"]
    if not isinstance(source_identity, Mapping) or set(source_identity) != {"git_commit", "sha256"}:
        raise ValueError("Expert 4 UniLM fingerprint must contain exactly git_commit and sha256")
    if source_identity["git_commit"] is not None and not isinstance(source_identity["git_commit"], str):
        raise ValueError("Expert 4 UniLM git_commit fingerprint must be a string or null")
    source_sha = source_identity["sha256"]
    if source_sha is not None and (not isinstance(source_sha, str) or not _SHA256_RE.fullmatch(source_sha)):
        raise ValueError("Expert 4 UniLM source sha256 must be a 64-character SHA-256 or null")

    if type(state["epoch"]) is not int or state["epoch"] < 0:
        raise ValueError("Expert 4 best checkpoint epoch must be a non-negative integer")
    best_srcc = state["best_srcc"]
    if isinstance(best_srcc, bool) or not isinstance(best_srcc, (float, int)) or not math.isfinite(float(best_srcc)):
        raise ValueError("Expert 4 best_srcc must be a finite number")
    return state


def _load_best_bundle_preflight(checkpoint_path: Path) -> Mapping[str, Any]:
    """Safely read and validate the source best artifact before model creation."""
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Expert 4 could not safely read fine-tuned best checkpoint {checkpoint_path}: {exc}. "
            "Only trusted artifacts in the source best-checkpoint format are supported."
        ) from exc
    return _validate_best_checkpoint_bundle(state)


def _feature_dim_from_model(model: nn.Module) -> Optional[int]:
    seq_coattn = getattr(model, "seq_coattn", None)
    dim = getattr(seq_coattn, "dim", None)
    fused_width = 2 * dim if type(dim) is int and dim > 0 else None
    head = getattr(model, "head", None)
    head_net = getattr(head, "net", None)
    head_first = head_net[0] if isinstance(head_net, (nn.Sequential, nn.ModuleList)) and len(head_net) else None
    head_width = getattr(head_first, "in_features", None)
    if head_width is not None and (type(head_width) is not int or head_width <= 0):
        raise ValueError(f"Expert 4 regression head exposes an invalid feature width: {head_width!r}")
    if fused_width is not None and head_width is not None and fused_width != head_width:
        raise ValueError(
            "Expert 4 fusion and regression-head metadata disagree: "
            f"2 * seq_coattn.dim={fused_width}, head input width={head_width}"
        )
    return fused_width if fused_width is not None else head_width


def _validate_restored_trainable_parameters(model: nn.Module, trained_state: Mapping[str, torch.Tensor]) -> None:
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected, actual = set(parameters), set(trained_state)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise ValueError(
            "Expert 4 restored trainable parameter set mismatch: "
            f"missing={sorted(missing)[:12]}, unexpected={sorted(unexpected)[:12]}"
        )
    shape_mismatches = [
        (name, tuple(parameters[name].shape), tuple(trained_state[name].shape))
        for name in sorted(expected)
        if tuple(parameters[name].shape) != tuple(trained_state[name].shape)
    ]
    if shape_mismatches:
        raise ValueError(f"Expert 4 fine-tuned parameter shape mismatch: {shape_mismatches[:8]}")
    value_mismatches = []
    for name in sorted(expected):
        loaded = parameters[name].detach().to(device="cpu")
        recorded = trained_state[name].detach().to(device="cpu", dtype=loaded.dtype)
        if not torch.equal(loaded, recorded):
            value_mismatches.append(name)
            if len(value_mismatches) >= 8:
                break
    if value_mismatches:
        raise ValueError(
            "Expert 4 original loader returned without restoring fine-tuned tensors exactly: "
            f"{value_mismatches}"
        )


class Expert4Adapter(ExpertAdapter):
    """Wrap the original fine-tuned BEATs/CLAP/SeqCoAttn Expert 4.

    Args:
        model: A restored original ``expert4_finetune.model.Expert4`` or a
            contract-compatible ``nn.Module`` used by lightweight tests.
        feature_dim: Optional trusted fused width. Production metadata derives
            this as ``2 * model.seq_coattn.dim`` and cross-checks the regression
            head's input width.

    Expert 4 returns score ``[B]`` and its unmodified SeqCoAttn fused vector.
    Only the score's singleton output axis is added to meet the common
    ``[B,1]`` contract. The wrapped model is frozen and held in eval mode.
    """

    def __init__(self, model: nn.Module, *, feature_dim: Optional[int] = None) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expert 4 model must be an nn.Module; got {type(model).__name__}")
        if feature_dim is not None and (type(feature_dim) is not int or feature_dim <= 0):
            raise ValueError(f"Expert 4 feature_dim must be a positive integer; got {feature_dim!r}")
        model_dim = _feature_dim_from_model(model)
        if feature_dim is None:
            feature_dim = model_dim
        elif model_dim is not None and feature_dim != model_dim:
            raise ValueError(
                "Expert 4 explicit feature_dim does not match fusion/head metadata: "
                f"{feature_dim} != {model_dim}"
            )
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError(
                "Expert 4 fused feature width is unavailable; construct the original model "
                "or pass a positive width from trusted source metadata"
            )
        super().__init__(feature_dim=feature_dim)
        self.model = model
        self._freeze_model()

    @classmethod
    def from_finetuned(
        cls,
        *,
        beats_checkpoint: PathValue,
        clap_checkpoint: PathValue,
        unilm_source_dir: PathValue,
        finetuned_checkpoint: PathValue,
        feature_dim: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        offline: bool = True,
    ) -> "Expert4Adapter":
        """Restore the source best bundle through its original loader.

        The source ``inference.load_finetuned`` checks SHA/source fingerprints,
        constructs ``Expert4.from_pretrained``, requires exact equality between
        fine-tuned keys and source-trainable parameters, then loads the partial
        state with the source's intentional ``strict=False`` semantics. This
        factory preflights the exact best bundle and verifies loaded values
        before freezing the complete model. The artifact must be trusted because
        the original loader uses ``weights_only=False`` for checkpoint loading.
        """
        target = _target_device(device)
        beats_path = _resolved_resource(beats_checkpoint, field="BEATs pretrained checkpoint")
        clap_path = _resolved_resource(clap_checkpoint, field="LAION-CLAP pretrained checkpoint")
        unilm_path = _resolved_resource(unilm_source_dir, field="UniLM/BEATs source", directory=True)
        finetuned_path = _resolved_resource(finetuned_checkpoint, field="fine-tuned best checkpoint")
        _beats_dir(unilm_path)
        bundle = _load_best_bundle_preflight(finetuned_path)

        # A conflicting top-level BEATs module (for example, one loaded by MGA)
        # is rejected before the original factory can retain the wrong class.
        validate_expert4_module_origins(unilm_path)
        resources = {
            "beats_checkpoint": beats_path,
            "clap_checkpoint": clap_path,
            "unilm_source_dir": unilm_path,
        }
        with _offline_huggingface(offline), _scoped_beats_import(unilm_path):
            try:
                original_inference = importlib.import_module("expert4_finetune.inference")
                model, restored_bundle = original_inference.load_finetuned(
                    finetuned_path, resources, target
                )
            except Exception as exc:
                raise RuntimeError(
                    "Expert 4 original fine-tuned restoration failed through "
                    "expert4_finetune.inference.load_finetuned; no pretrained-only fallback was used"
                ) from exc

            _validate_best_checkpoint_bundle(restored_bundle)
            # Ensure the original source loader used the same validated artifact
            # and did not return a different state between preflight and restore.
            if set(restored_bundle["model_state_dict"]) != set(bundle["model_state_dict"]):
                raise ValueError("Expert 4 checkpoint trained-key set changed during restoration")
            for name in bundle["model_state_dict"]:
                if not torch.equal(restored_bundle["model_state_dict"][name], bundle["model_state_dict"][name]):
                    raise ValueError(f"Expert 4 checkpoint tensor changed during restoration: {name}")
            if not isinstance(model, nn.Module):
                raise TypeError("Expert 4 original loader did not return an nn.Module")
            _validate_restored_trainable_parameters(model, restored_bundle["model_state_dict"])
            validate_expert4_module_origins(unilm_path, require_beats=True)

        adapter = cls(model, feature_dim=feature_dim)
        return adapter.to(target).eval()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Expert4Adapter":
        """Construct from LOCAL-03 full resolved config or its Expert 4 section."""
        if not isinstance(config, Mapping):
            raise TypeError(f"Expert 4 config must be a mapping; got {type(config).__name__}")
        if "experts" in config:
            experts = config.get("experts")
            if not isinstance(experts, Mapping) or "expert4" not in experts:
                raise ValueError("Expert 4 config is missing experts.expert4")
            section = experts["expert4"]
            runtime = config.get("runtime", {})
            if not isinstance(runtime, Mapping):
                raise TypeError("Expert 4 config runtime must be a mapping")
        else:
            section, runtime = config, {}
        if not isinstance(section, Mapping):
            raise TypeError("Expert 4 resource config must be a mapping")
        required = {
            "beats_checkpoint",
            "clap_checkpoint",
            "unilm_source_dir",
            "finetuned_checkpoint",
        }
        missing = required - set(section)
        if missing:
            raise ValueError(f"Expert 4 config is missing resource(s): {sorted(missing)}")
        selected_device = device if device is not None else runtime.get("device", "cpu")
        return cls.from_finetuned(
            beats_checkpoint=section["beats_checkpoint"],
            clap_checkpoint=section["clap_checkpoint"],
            unilm_source_dir=section["unilm_source_dir"],
            finetuned_checkpoint=section["finetuned_checkpoint"],
            device=selected_device,
            offline=runtime.get("offline", True),
        )

    def _freeze_model(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "Expert4Adapter":
        super().train(mode)
        self._freeze_model()
        return self

    def _model_device(self) -> Optional[torch.device]:
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        if not tensors:
            return None
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError(f"Expert 4 wrapped model spans multiple devices: {sorted(map(str, devices))}")
        return next(iter(devices))

    def _validate_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expert 4 batch must be a mapping; got {type(batch).__name__}")
        for key in ("expert4_wavs", "expert4_wav_lens", "captions_raw"):
            if key not in batch:
                raise KeyError(f"Expert 4 batch is missing required field {key!r}")
        wavs = batch["expert4_wavs"]
        lengths = batch["expert4_wav_lens"]
        captions = batch["captions_raw"]
        if not isinstance(wavs, torch.Tensor):
            raise TypeError("Expert 4 expert4_wavs must be a torch.Tensor")
        if wavs.ndim != 3 or wavs.shape[1] != 1:
            raise ValueError(
                "Expert 4 expert4_wavs must have rank 3 with a single audio channel "
                f"and shape [B,1,T]; got {tuple(wavs.shape)}"
            )
        batch_size, _, sample_count = wavs.shape
        if batch_size <= 0:
            raise ValueError("Expert 4 waveform batch must be non-empty")
        if sample_count <= 0 or sample_count > _MAX_AUDIO_SAMPLES:
            raise ValueError(
                f"Expert 4 waveform time axis must contain 1..{_MAX_AUDIO_SAMPLES} samples; "
                f"got {sample_count}"
            )
        if wavs.dtype != torch.float32:
            raise TypeError(f"Expert 4 waveforms must be float32; got {wavs.dtype}")
        if not torch.isfinite(wavs).all().item():
            raise ValueError("Expert 4 waveforms must contain only finite values")
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("Expert 4 expert4_wav_lens must be a torch.Tensor")
        if lengths.shape != (batch_size,):
            raise ValueError(
                f"Expert 4 waveform lengths must have shape [B]=[{batch_size}]; "
                f"got {tuple(lengths.shape)}"
            )
        if lengths.dtype != torch.int64:
            raise TypeError(f"Expert 4 waveform lengths must use torch.int64; got {lengths.dtype}")
        if lengths.device != wavs.device:
            raise ValueError(
                f"Expert 4 waveform and length tensors must share a device; got {wavs.device} and {lengths.device}"
            )
        if (lengths <= 0).any().item():
            raise ValueError("Expert 4 valid waveform lengths must be positive")
        if (lengths > sample_count).any().item():
            raise ValueError("Expert 4 valid waveform length exceeds padded waveform width")
        if (lengths > _MAX_AUDIO_SAMPLES).any().item():
            raise ValueError(f"Expert 4 valid waveform lengths cannot exceed {_MAX_AUDIO_SAMPLES} samples")

        if isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence):
            raise TypeError("Expert 4 captions_raw must be a sequence of raw strings")
        if len(captions) != batch_size:
            raise ValueError(
                f"Expert 4 caption count must match waveform batch size {batch_size}; got {len(captions)}"
            )
        if any(not isinstance(caption, str) for caption in captions):
            raise TypeError("Every Expert 4 caption must be a string")
        normalized_captions = [caption.strip() for caption in captions]
        if any(not caption for caption in normalized_captions):
            raise ValueError("Expert 4 captions must be non-empty after the original whitespace strip")

        model_device = self._model_device()
        if model_device is not None and wavs.device != model_device:
            raise ValueError(
                f"Expert 4 inputs are on {wavs.device}, but wrapped model is on {model_device}"
            )
        return wavs, lengths, normalized_captions

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        wavs, lengths, captions = self._validate_batch(batch)
        original_batch = {"wavs": wavs, "wav_lens": lengths, "captions": captions}
        output = self.model(original_batch)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("Original Expert 4 forward must return (score, fused_features)")
        score, features = output
        if not isinstance(score, torch.Tensor):
            raise TypeError("Original Expert 4 score must be a torch.Tensor")
        if score.ndim == 1 and score.shape[0] == wavs.shape[0]:
            score = score.unsqueeze(-1)
        elif score.ndim != 2 or tuple(score.shape) != (wavs.shape[0], 1):
            raise ValueError(
                "Original Expert 4 score must have shape [B] or [B,1]; "
                f"got {tuple(score.shape)}"
            )
        return score, features


__all__ = ["Expert4Adapter", "validate_expert4_module_origins"]
