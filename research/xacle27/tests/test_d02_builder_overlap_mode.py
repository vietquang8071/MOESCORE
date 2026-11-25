import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from data.full8_assembler import build_full8_stores
from scripts.build_full7_feature_store import main as build_full7_main


class D02BuilderOverlapModeTests(unittest.TestCase):
    def test_full8_skip_mode_builds_all_three_requested_splits(self):
        requested = ("train", "validation", "test")
        with (
            patch("data.full8_assembler.validate_feature_store",
                  side_effect=lambda root, *, split, require_labels=None: {"split": split}) as validate,
            patch("data.full8_assembler.build_full8_split",
                  side_effect=lambda *args, split, **kwargs: {"split": split}) as build,
            patch("data.audio_overlap.build_train_validation_audio_overlap_diagnostic",
                  side_effect=AssertionError("cross-split audio overlap was checked")),
            patch("data.audio_overlap.build_public_split_audio_overlap_diagnostic",
                  side_effect=AssertionError("cross-split audio overlap was checked")),
        ):
            report = build_full8_stores(
                "full7", "e8", "full8", splits=requested,
                skip_cross_split_audio_overlap_check=True,
            )

        self.assertEqual([call.kwargs["split"] for call in validate.call_args_list], list(requested))
        self.assertEqual([call.kwargs["split"] for call in build.call_args_list], list(requested))
        self.assertEqual(report["cross_split_audio_overlap"]["scope"], list(requested))

    def test_full8_skip_mode_needs_only_requested_full7_splits_and_skips_diagnostics(self):
        requested = ("train", "validation")

        def validate_store(root, *, split, require_labels=None):
            if split not in requested:
                raise FileNotFoundError(f"unexpected split lookup: {split}")
            return {"split": split}

        with (
            patch("data.full8_assembler.validate_feature_store", side_effect=validate_store) as validate,
            patch("data.full8_assembler.build_full8_split",
                  side_effect=lambda *args, split, **kwargs: {"split": split}) as build,
            patch("data.audio_overlap.build_train_validation_audio_overlap_diagnostic",
                  side_effect=AssertionError("cross-split audio overlap was checked")),
            patch("data.audio_overlap.build_public_split_audio_overlap_diagnostic",
                  side_effect=AssertionError("cross-split audio overlap was checked")),
        ):
            report = build_full8_stores(
                "full7", "e8", "full8", splits=requested,
                skip_cross_split_audio_overlap_check=True,
            )

        self.assertEqual([call.kwargs["split"] for call in validate.call_args_list], list(requested))
        self.assertEqual([call.kwargs["split"] for call in build.call_args_list], list(requested))
        self.assertEqual(report["cross_split_audio_overlap"], {
            "status": "SKIPPED",
            "reason": "Explicit D02 research-only configuration",
            "scope": list(requested),
        })


    def test_full7_cli_skip_mode_builds_three_requested_splits_without_overlap_audits(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            requested = ("train", "validation", "test")
            output = io.StringIO()
            arguments = [
                "--full9-root", str(root / "full9"),
                "--whisper-root", str(root / "whisper"),
                "--e5-checkpoint", str(root / "e5.pt"),
                "--e6-checkpoint", str(root / "e6.pt"),
                "--output-root", str(root / "full7"),
                "--source-audio-root", str(root / "wav"),
                "--skip-cross-split-audio-overlap-check",
                *(item for split in requested for item in ("--split", split)),
            ]
            with (
                patch("scripts.build_full7_feature_store.build_full7_split",
                      side_effect=lambda *args, split, **kwargs: {"split": split}) as build,
                patch("scripts.build_full7_feature_store.validate_feature_store",
                      side_effect=AssertionError("unrequested source manifest was loaded")),
                patch("scripts.build_full7_feature_store.build_train_validation_audio_overlap_diagnostic",
                      side_effect=AssertionError("cross-split audio overlap was checked")),
                patch("scripts.build_full7_feature_store.build_public_split_audio_overlap_diagnostic",
                      side_effect=AssertionError("cross-split audio overlap was checked")),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(build_full7_main(arguments), 0)

            report = json.loads(output.getvalue())
            self.assertEqual([call.kwargs["split"] for call in build.call_args_list], list(requested))
            self.assertEqual(report["cross_split_audio_overlap"], {
                "status": "SKIPPED",
                "reason": "Explicit D02 research-only configuration",
                "scope": list(requested),
            })
