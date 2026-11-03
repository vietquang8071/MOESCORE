import copy
import tempfile
import unittest
from pathlib import Path

from configs.moe_config import (
    MoeConfigError,
    load_moe_config,
    resolve_moe_paths,
    validate_moe_config,
    validate_moe_resources,
)


def valid_config():
    return {
        "schema_version": 1,
        "runtime": {"device": "auto", "offline": True, "cache_dir": ".cache/hf"},
        "paths": {"project_root": "..", "data_root": "../data", "resource_root": ".."},
        "data": {
            "train_csv": "train.csv",
            "validation_csv": "validation.csv",
            "audio_root": "audio",
        },
        "experts": {
            "expert1": {
                "pretrained_checkpoint": "weights/clap.pt",
                "finetuned_checkpoint": "weights/expert1_best.pt",
                "enable_fusion": False,
            },
            "expert2": {
                "source_dir": "expert2/MGA-CLAP",
                "architecture_yaml": "expert2/settings/model.yaml",
                "training_config": "expert2/train.yaml",
                "pretrained_checkpoint": "weights/mga.pt",
                "finetuned_checkpoint": "weights/expert2_best.pt",
                "freeze_policy": "partial",
            },
            "expert3": {
                "source_dir": "expert3/source",
                "base_checkpoint": "weights/m2d.pth",
                "expected_base_sha256": "a" * 64,
                "finetuned_checkpoint": "weights/expert3_best.pt",
                "concat_assignment": "released",
                "model_config": "expert3/config.json",
            },
            "expert4": {
                "beats_checkpoint": "weights/beats.pt",
                "clap_checkpoint": "weights/clap.pt",
                "unilm_source_dir": "expert4/unilm",
                "finetuned_checkpoint": "weights/expert4_best.pt",
            },
        },
        "gate": {
            "expert_order": ["expert1", "expert2", "expert3", "expert4"],
            "hidden_dim": 512,
            "dropout": 0.1,
        },
    }


class MoeConfigSchemaTests(unittest.TestCase):
    def test_valid_config_mapping_passes_schema_validation(self):
        validate_moe_config(valid_config())

    def test_example_config_loads_and_is_schema_valid_without_resources(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "moe_local.example.yaml"
        config = load_moe_config(path)
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(
            config["experts"]["expert3"]["expected_base_sha256"],
            "238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1",
        )
        self.assertEqual(
            config["experts"]["expert4"]["unilm_source_dir"],
            "/path/to/unilm_expert4_original",
        )

    def test_unsupported_schema_version_is_rejected(self):
        config = valid_config()
        config["schema_version"] = 2
        with self.assertRaisesRegex(MoeConfigError, "schema_version"):
            validate_moe_config(config)

    def test_missing_required_section_or_expert_is_rejected(self):
        config = valid_config()
        del config["data"]
        with self.assertRaisesRegex(MoeConfigError, "data"):
            validate_moe_config(config)

        config = valid_config()
        del config["experts"]["expert3"]
        with self.assertRaisesRegex(MoeConfigError, "expert3"):
            validate_moe_config(config)

    def test_unknown_fields_are_rejected(self):
        config = valid_config()
        config["experts"]["expert1"]["fallback_to_pretrained"] = True
        with self.assertRaisesRegex(MoeConfigError, "fallback_to_pretrained"):
            validate_moe_config(config)

    def test_exact_expert_set_and_order_are_required(self):
        config = valid_config()
        config["experts"]["expert5"] = {}
        with self.assertRaisesRegex(MoeConfigError, "exactly expert1.*expert4"):
            validate_moe_config(config)

        config = valid_config()
        config["gate"]["expert_order"] = ["expert2", "expert1", "expert3", "expert4"]
        with self.assertRaisesRegex(MoeConfigError, "expert_order"):
            validate_moe_config(config)

    def test_gate_dimensions_and_dropout_are_validated(self):
        for key, value in (("hidden_dim", 0), ("hidden_dim", True), ("dropout", -0.1), ("dropout", 1.0)):
            config = valid_config()
            config["gate"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(MoeConfigError, "gate"):
                    validate_moe_config(config)

    def test_runtime_types_and_device_are_validated(self):
        for key, value in (("device", "tpu"), ("offline", "true"), ("cache_dir", 7)):
            config = valid_config()
            config["runtime"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(MoeConfigError, "runtime"):
                    validate_moe_config(config)
        config = valid_config()
        config["runtime"]["device"] = "cuda:2"
        validate_moe_config(config)

    def test_path_fields_must_be_nonempty_strings_or_null_when_optional(self):
        config = valid_config()
        config["data"]["train_csv"] = 12
        with self.assertRaisesRegex(MoeConfigError, "data.train_csv"):
            validate_moe_config(config)

        config = valid_config()
        config["runtime"]["cache_dir"] = ""
        with self.assertRaisesRegex(MoeConfigError, "runtime.cache_dir"):
            validate_moe_config(config)

    def test_expert_specific_values_and_sha_format_are_validated(self):
        config = valid_config()
        config["experts"]["expert1"]["enable_fusion"] = True
        with self.assertRaisesRegex(MoeConfigError, "enable_fusion"):
            validate_moe_config(config)

        config = valid_config()
        config["experts"]["expert2"]["freeze_policy"] = "none"
        with self.assertRaisesRegex(MoeConfigError, "freeze_policy"):
            validate_moe_config(config)

        config = valid_config()
        config["experts"]["expert2"]["freeze_policy"] = ["partial"]
        with self.assertRaisesRegex(MoeConfigError, "freeze_policy"):
            validate_moe_config(config)

        config = valid_config()
        config["experts"]["expert3"]["concat_assignment"] = "semantic-ish"
        with self.assertRaisesRegex(MoeConfigError, "concat_assignment"):
            validate_moe_config(config)

        config = valid_config()
        config["experts"]["expert3"]["concat_assignment"] = []
        with self.assertRaisesRegex(MoeConfigError, "concat_assignment"):
            validate_moe_config(config)

        config = valid_config()
        config["experts"]["expert3"]["expected_base_sha256"] = "REQUIRED"
        with self.assertRaisesRegex(MoeConfigError, "64 hexadecimal"):
            validate_moe_config(config)

        config = valid_config()
        config["experts"]["expert3"]["expected_base_sha256"] = None
        validate_moe_config(config)

    def test_safe_yaml_loader_rejects_python_object_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.yaml"
            path.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
            with self.assertRaises(MoeConfigError):
                load_moe_config(path)


class MoeConfigPathTests(unittest.TestCase):
    def test_relative_paths_default_to_config_relative_roots(self):
        config = valid_config()
        config["paths"] = {"project_root": None, "data_root": None, "resource_root": None}
        config_file = Path("/tmp/moescore config/config.yaml")
        resolved = resolve_moe_paths(config, config_file, environ={})
        config_dir = config_file.resolve().parent
        self.assertEqual(resolved["paths"]["project_root"], config_dir)
        self.assertEqual(resolved["data"]["train_csv"], config_dir / "train.csv")
        self.assertEqual(
            resolved["experts"]["expert1"]["pretrained_checkpoint"],
            config_dir / "weights" / "clap.pt",
        )
        self.assertEqual(resolved["runtime"]["cache_dir"], config_dir / ".cache" / "hf")

    def test_explicit_roots_precede_environment_and_config_roots(self):
        config = valid_config()
        with tempfile.TemporaryDirectory(prefix="moescore path test ") as temporary:
            root = Path(temporary)
            resolved = resolve_moe_paths(
                config,
                root / "configs" / "moe.yaml",
                project_root=root / "explicit project",
                data_root=root / "explicit data",
                resource_root=root / "explicit resources",
                environ={
                    "MOESCORE_ROOT": str(root / "env project"),
                    "MOESCORE_DATA_ROOT": str(root / "env data"),
                    "MOESCORE_RESOURCE_ROOT": str(root / "env resources"),
                },
            )
            self.assertEqual(resolved["paths"]["project_root"], root / "explicit project")
            self.assertEqual(resolved["data"]["train_csv"], root / "explicit data" / "train.csv")
            self.assertEqual(
                resolved["experts"]["expert1"]["pretrained_checkpoint"],
                root / "explicit resources" / "weights" / "clap.pt",
            )

    def test_environment_roots_precede_config_roots(self):
        config = valid_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = resolve_moe_paths(
                config,
                root / "configs" / "moe.yaml",
                environ={
                    "MOESCORE_ROOT": str(root / "env project"),
                    "MOESCORE_DATA_ROOT": str(root / "env data"),
                    "MOESCORE_RESOURCE_ROOT": str(root / "env resources"),
                },
            )
            self.assertEqual(resolved["paths"]["project_root"], root / "env project")
            self.assertEqual(resolved["data"]["validation_csv"], root / "env data" / "validation.csv")
            self.assertEqual(
                resolved["experts"]["expert3"]["base_checkpoint"],
                root / "env resources" / "weights" / "m2d.pth",
            )

    def test_config_roots_precede_project_default_and_config_file_relative_paths(self):
        config = valid_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["paths"] = {
                "project_root": "project root",
                "data_root": "dataset root",
                "resource_root": "resource root",
            }
            resolved = resolve_moe_paths(config, root / "configs" / "moe.yaml", environ={})
            self.assertEqual(resolved["paths"]["project_root"], root / "configs" / "project root")
            self.assertEqual(resolved["data"]["audio_root"], root / "configs" / "dataset root" / "audio")
            self.assertEqual(
                resolved["experts"]["expert4"]["unilm_source_dir"],
                root / "configs" / "resource root" / "expert4" / "unilm",
            )

    def test_explicitly_absolute_resource_paths_are_not_rebased(self):
        config = valid_config()
        absolute = Path("/tmp/checkpoint with spaces/expert.pt")
        config["experts"]["expert1"]["pretrained_checkpoint"] = str(absolute)
        resolved = resolve_moe_paths(config, "/tmp/config.yaml", resource_root="/tmp/other", environ={})
        self.assertEqual(
            resolved["experts"]["expert1"]["pretrained_checkpoint"], absolute
        )

    def test_path_resolution_does_not_require_or_open_resources(self):
        config = valid_config()
        with tempfile.TemporaryDirectory() as temporary:
            resolved = resolve_moe_paths(config, Path(temporary) / "moe.yaml", environ={})
            self.assertFalse(resolved["experts"]["expert1"]["finetuned_checkpoint"].exists())

    def test_path_resolution_does_not_mutate_input_mapping(self):
        config = valid_config()
        original = copy.deepcopy(config)
        resolve_moe_paths(config, "/tmp/moe.yaml", environ={})
        self.assertEqual(config, original)

    def test_optional_test_csv_is_resolved_when_present_and_remains_backward_compatible(self):
        config = valid_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["data"]["test_csv"] = "metadata/test_average.csv"
            resolved = resolve_moe_paths(
                config, root / "configs" / "moe.yaml", data_root=root / "data", environ={}
            )
            self.assertEqual(resolved["data"]["test_csv"], root / "data" / "metadata" / "test_average.csv")
            old_schema = valid_config()
            self.assertNotIn("test_csv", old_schema["data"])
            validate_moe_config(old_schema)
            old_resolved = resolve_moe_paths(old_schema, root / "configs" / "moe.yaml", environ={})
            self.assertIsNone(old_resolved["data"]["test_csv"])

    def test_strict_resource_validation_reports_missing_resources_and_sha(self):
        config = valid_config()
        config["experts"]["expert3"]["expected_base_sha256"] = None
        resolved = resolve_moe_paths(config, "/tmp/moescore config/moe.yaml", environ={})
        with self.assertRaises(MoeConfigError) as context:
            validate_moe_resources(resolved)
        message = str(context.exception)
        self.assertIn("experts.expert1.finetuned_checkpoint", message)
        self.assertIn("expected resource type", message)
        self.assertIn("experts.expert3.expected_base_sha256", message)
        self.assertIn("/tmp/weights/expert1_best.pt", message)

    def test_existing_resource_fixtures_pass_strict_validation(self):
        config = valid_config()
        config["experts"]["expert3"]["expected_base_sha256"] = "b" * 64
        with tempfile.TemporaryDirectory(prefix="moescore resources ") as temporary:
            root = Path(temporary)
            resolved = resolve_moe_paths(
                config,
                root / "configs" / "moe.yaml",
                project_root=root,
                data_root=root / "data",
                resource_root=root / "resources",
                environ={},
            )
            for field in ("train_csv", "validation_csv"):
                path = resolved["data"][field]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            directory_fields = (
                ("data", "audio_root"),
                ("experts", "expert2", "source_dir"),
                ("experts", "expert3", "source_dir"),
                ("experts", "expert4", "unilm_source_dir"),
            )
            for keys in directory_fields:
                node = resolved
                for key in keys:
                    node = node[key]
                node.mkdir(parents=True, exist_ok=True)
            file_paths = [
                resolved["experts"]["expert1"]["pretrained_checkpoint"],
                resolved["experts"]["expert1"]["finetuned_checkpoint"],
                resolved["experts"]["expert2"]["architecture_yaml"],
                resolved["experts"]["expert2"]["training_config"],
                resolved["experts"]["expert2"]["pretrained_checkpoint"],
                resolved["experts"]["expert2"]["finetuned_checkpoint"],
                resolved["experts"]["expert3"]["base_checkpoint"],
                resolved["experts"]["expert3"]["finetuned_checkpoint"],
                resolved["experts"]["expert3"]["model_config"],
                resolved["experts"]["expert4"]["beats_checkpoint"],
                resolved["experts"]["expert4"]["clap_checkpoint"],
                resolved["experts"]["expert4"]["finetuned_checkpoint"],
            ]
            for path in file_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            validate_moe_resources(resolved)

    def test_missing_finetuned_checkpoint_never_falls_back_to_pretrained(self):
        config = valid_config()
        resolved = resolve_moe_paths(config, "/tmp/config.yaml", environ={})
        with self.assertRaises(MoeConfigError) as context:
            validate_moe_resources(resolved)
        self.assertIn("expert1.finetuned_checkpoint", str(context.exception))
        self.assertIn("expert2.finetuned_checkpoint", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
