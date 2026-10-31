import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf


class ScalarBlock(nn.Module):
    def __init__(self, value=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return x + self.scale


class FakePortableM2D(nn.Module):
    def __init__(self, weight_file, flat_features=True, dim=768, with_text_proj=True):
        super().__init__()
        self.weight_file = weight_file
        self.flat_features = flat_features
        self.text_ready = False
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Linear(dim, dim)
        self.backbone.blocks = nn.ModuleList([ScalarBlock(i / 1000) for i in range(12)])
        self.backbone.audio_proj = nn.Linear(dim, dim)
        bert = nn.Module()
        bert.config = SimpleNamespace(hidden_size=dim)
        bert.encoder = nn.Module()
        bert.encoder.layer = nn.ModuleList([ScalarBlock(i / 1000) for i in range(12)])
        self.text_encoder = nn.Module()
        self.text_encoder.text_encoder = bert
        if with_text_proj:
            self.backbone.text_proj = nn.Linear(dim, dim)

    def get_clap_text_encoder(self):
        self.text_ready = True

    def encode_clap_audio(self, wavs):
        x = wavs.mean(dim=tuple(range(1, wavs.ndim)), keepdim=False).unsqueeze(-1).expand(-1, 768)
        x = self.backbone.patch_embed(x)
        for block in self.backbone.blocks:
            x = block(x)
        return self.backbone.audio_proj(x)

    def encode_clap_text(self, captions):
        device = next(self.parameters()).device
        x = torch.tensor([len(s) / 100 for s in captions], dtype=torch.float32, device=device).unsqueeze(-1).expand(-1, 768)
        for block in self.text_encoder.text_encoder.encoder.layer:
            x = block(x)
        return self.backbone.text_proj(x) if hasattr(self.backbone, "text_proj") else x


class Expert3ModelTest(unittest.TestCase):
    def make_model(self, root, activation_checkpointing=True, with_text_proj=True):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import MoescoreExpert3

        source = root / "checkpoint-30.pth"
        source.write_bytes(b"test base checkpoint")
        fake = FakePortableM2D(str(source), with_text_proj=with_text_proj)
        with patch("models.moescore_expert3._load_portable_m2d", return_value=fake):
            model = MoescoreExpert3(
                checkpoint_path=source,
                cache_dir=root / "hf_cache",
                activation_checkpointing=activation_checkpointing,
            )
        return model, fake, source

    def test_checkpoint_without_text_projector_uses_bert_output_and_keeps_bert_trainable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, fake, _ = self.make_model(root, with_text_proj=False)
            self.assertFalse(hasattr(fake.backbone, "text_proj"))
            self.assertEqual(model.embedding_dim, 768)
            self.assertTrue(fake.backbone.audio_proj.weight.requires_grad)
            bert_layers = fake.text_encoder.text_encoder.encoder.layer
            self.assertTrue(bert_layers[-1].scale.requires_grad)
            self.assertTrue(bert_layers[-2].scale.requires_grad)
            score, cross = model({"wavs": torch.randn(8, 1, 16000), "captions": ["caption"] * 8})
            self.assertEqual(tuple(score.shape), (8, 1))
            self.assertEqual(tuple(cross.shape), (8, 3840))

    def test_regression_head_stays_float32_under_outer_autocast(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.losses.loss_function import get_loss_function

        with tempfile.TemporaryDirectory() as tmp:
            model, _, _ = self.make_model(Path(tmp), activation_checkpointing=False, with_text_proj=False)
            model.train()
            batch = {
                "wavs": torch.randn(8, 1, 16000),
                "captions": ["a sound caption"] * 8,
            }
            target = torch.linspace(-0.8, 0.8, 8)
            num_class = torch.ones(8, dtype=torch.long)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                prediction, _ = model(batch)
            loss_fn = get_loss_function({
                "loss_weights": [1.0, 0.5],
                "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
                "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
            })
            loss = loss_fn(prediction.reshape(-1).float(), target, num_class)
            self.assertEqual(prediction.dtype, torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(model.alignment_head[-1].bias.grad).all())

    def test_checkpoint_path_is_wrapped_with_expected_model_parent_without_copying(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import prepare_m2d_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "checkpoint-30.pth"
            original.write_bytes(b"immutable pretrained bytes")
            expected = prepare_m2d_checkpoint(original, root / "layout")
            self.assertEqual(expected.parent.name, "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025")
            self.assertTrue(expected.is_symlink())
            self.assertEqual(expected.read_bytes(), original.read_bytes())

    def test_initializes_text_then_unfreezes_only_requested_modules_and_emits_released_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, fake, _ = self.make_model(Path(tmp))
            self.assertTrue(fake.text_ready)
            self.assertFalse(fake.backbone.patch_embed.weight.requires_grad)
            self.assertFalse(fake.backbone.blocks[9].scale.requires_grad)
            self.assertTrue(fake.backbone.blocks[10].scale.requires_grad)
            self.assertTrue(fake.backbone.blocks[11].scale.requires_grad)
            layers = fake.text_encoder.text_encoder.encoder.layer
            self.assertFalse(layers[9].scale.requires_grad)
            self.assertTrue(layers[10].scale.requires_grad)
            self.assertTrue(layers[11].scale.requires_grad)
            self.assertTrue(fake.backbone.audio_proj.weight.requires_grad)
            self.assertTrue(fake.backbone.text_proj.weight.requires_grad)

            wavs = torch.randn(8, 1, 16000)
            captions = [f"caption {i}" for i in range(8)]
            with torch.no_grad():
                text_emb = F.normalize(fake.encode_clap_text(captions), dim=-1)
            score, cross = model({"wavs": wavs, "captions": captions})
            self.assertEqual(tuple(score.shape), (8, 1))
            self.assertEqual(tuple(cross.shape), (8, 3840))
            self.assertTrue(torch.allclose(cross[:, :768], text_emb, atol=1e-6))

    def test_batch_eight_backward_updates_trainable_modules_only_with_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, fake, _ = self.make_model(Path(tmp), activation_checkpointing=True)
            model.train()
            score, cross = model({"wavs": torch.randn(8, 1, 16000), "captions": ["abc"] * 8})
            self.assertEqual(tuple(score.shape), (8, 1))
            loss = score.float().square().mean() + cross.float().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNone(fake.backbone.blocks[9].scale.grad)
            self.assertIsNotNone(fake.backbone.blocks[10].scale.grad)
            self.assertIsNotNone(fake.text_encoder.text_encoder.encoder.layer[10].scale.grad)
            self.assertIsNotNone(fake.backbone.audio_proj.weight.grad)
            self.assertIsNotNone(fake.backbone.text_proj.weight.grad)
            self.assertIsNotNone(model.alignment_head[0].weight.grad)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class Expert3CheckpointAndLossTest(unittest.TestCase):
    def test_best_and_resume_checkpoint_restore_trainables_and_training_state(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import MoescoreExpert3
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.utils.expert3_checkpoint import (
            load_best_checkpoint,
            load_resume_checkpoint,
            save_best_checkpoint,
            save_resume_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "checkpoint-30.pth"
            source.write_bytes(b"base")
            fake = FakePortableM2D(str(source))
            with patch("models.moescore_expert3._load_portable_m2d", return_value=fake):
                model = MoescoreExpert3(source, cache_dir=root / "cache")
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            model.eval()
            example = {"wavs": torch.ones(2, 1, 16000), "captions": ["a", "longer caption"]}
            expected_prediction = model(example)[0].detach().clone()
            before = model.alignment_head[0].weight.detach().clone()
            best = root / "best_model.pt"
            save_best_checkpoint(model, best, epoch=3, metrics={"SRCC": 0.5})
            with torch.no_grad():
                model.alignment_head[0].weight.add_(1.0)
            load_best_checkpoint(model, best)
            self.assertTrue(torch.equal(model.alignment_head[0].weight, before))
            self.assertTrue(torch.equal(model(example)[0], expected_prediction))

            resume = root / "last_checkpoint.pt"
            save_resume_checkpoint(model, resume, optimizer, scheduler, scaler, epoch=4,
                                   best_srcc=0.5, stale_epochs=1, config={"batch_size": 8})
            with torch.no_grad():
                model.alignment_head[0].weight.zero_()
            restored = load_resume_checkpoint(model, resume, optimizer, scheduler, scaler)
            self.assertTrue(torch.equal(model.alignment_head[0].weight, before))
            self.assertEqual(restored["epoch"], 4)
            self.assertEqual(restored["stale_epochs"], 1)

    def test_baseline_hybrid_loss_uses_eight_flat_examples_and_tau_moves_with_device(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.losses.loss_function import get_loss_function

        cfg = {
            "loss_weights": [1.0, 0.5],
            "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
            "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
        }
        loss_fn = get_loss_function(cfg)
        self.assertTrue(any(name.endswith(".tau") for name, _ in loss_fn.named_buffers()))
        pred = torch.tensor([-0.8, -0.5, -0.1, 0.05, 0.25, 0.45, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([-0.75, -0.4, -0.2, 0.0, 0.3, 0.4, 0.65, 0.85])
        got = loss_fn(pred.reshape(8, 1).reshape(-1), target.reshape(-1), torch.ones(8, dtype=torch.long))
        clipped = (((pred - target).square()) * ((pred - target).abs() > 0.25)).mean()
        target_diff = target[:, None] - target[None, :]
        pred_diff = pred[:, None] - pred[None, :]
        pairwise = (torch.relu((pred_diff - target_diff).abs() - 0.1).mean()) / 2
        expected = clipped + 0.5 * pairwise
        self.assertTrue(torch.allclose(got, expected))
        got.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        if torch.cuda.is_available():
            gpu_loss_fn = loss_fn.to("cuda")
            self.assertEqual(dict(gpu_loss_fn.named_buffers())["loss_instances.0.tau"].device.type, "cuda")
            gpu_pred = pred.detach().cuda().requires_grad_(True)
            gpu_value = gpu_loss_fn(gpu_pred, target.cuda(), torch.ones(8, dtype=torch.long, device="cuda"))
            gpu_value.backward()
            self.assertTrue(torch.isfinite(gpu_value))
            self.assertTrue(torch.isfinite(gpu_pred.grad).all())


class Expert3DataAndConfigTest(unittest.TestCase):
    def _write_split(self, root, split, count, sample_rate=16000, channels=1):
        csv_path = root / "meta_data" / f"{split}_average.csv"
        wav_dir = root / "wav" / split
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            import csv
            writer = csv.writer(stream)
            writer.writerow(("wav_file_name", "text", "average_score"))
            for i in range(count):
                filename = f"{split}_{i}.wav"
                audio = np.full((sample_rate // 10, channels), 0.1, dtype=np.float32)
                sf.write(wav_dir / filename, audio, sample_rate)
                writer.writerow((filename, f"sample caption {i}", 2.5 + (i % 5) * 0.5))

    def test_loader_resamples_downmixes_pads_and_normalizes_scores(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.datasets.moescore_expert3_dataset import MoescoreExpert3Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_split(root, "train", 2, sample_rate=8000, channels=2)
            dataset = MoescoreExpert3Dataset(
                root / "meta_data/train_average.csv", root / "wav/train", sample_rate=16000, max_len=10
            )
            item = dataset[0]
            self.assertEqual(tuple(item["wav"].shape), (1, 160000))
            self.assertAlmostEqual(float(item["score"]), -0.5)
            batch = dataset.collate_fn([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["wavs"].shape), (2, 1, 160000))
            self.assertEqual(tuple(batch["scores"].shape), (2,))
            self.assertTrue(all(isinstance(text, str) for text in batch["captions"]))

    def test_unlabeled_inference_pairs_produce_audio_and_caption_without_fake_targets(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.datasets.moescore_expert3_dataset import MoescoreExpert3Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_dir = root / "wav"
            wav_dir.mkdir()
            sf.write(wav_dir / "unlabeled.wav", np.zeros(16000, dtype=np.float32), 16000)
            metadata = root / "unlabeled.csv"
            with metadata.open("w", newline="", encoding="utf-8") as stream:
                import csv
                writer = csv.writer(stream)
                writer.writerow(("wav_file_name", "text"))
                writer.writerow(("unlabeled.wav", "sound of rain"))
            dataset = MoescoreExpert3Dataset(metadata, wav_dir, require_scores=False)
            batch = dataset.collate_fn([dataset[0]])
            self.assertEqual(tuple(batch["wavs"].shape), (1, 1, 160000))
            self.assertEqual(batch["captions"], ["sound of rain"])
            self.assertNotIn("scores", batch)

    def test_custom_inference_writes_score_csv_for_one_unlabeled_pair(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.inference_expert3 import run_inference
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.train_expert3 import load_config

        class StubModel(nn.Module):
            def forward(self, batch):
                self.asserted_captions = batch["captions"]
                return torch.full((len(batch["captions"]), 1), 0.2), torch.empty(0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            sf.write(audio_dir / "one.wav", np.zeros(16000, dtype=np.float32), 16000)
            metadata = root / "pairs.csv"
            with metadata.open("w", newline="", encoding="utf-8") as stream:
                import csv
                writer = csv.writer(stream)
                writer.writerow(("wav_file_name", "text"))
                writer.writerow(("one.wav", "a single caption"))
            base_checkpoint = root / "checkpoint-30.pth"
            base_checkpoint.write_bytes(b"placeholder")
            best_checkpoint = root / "best_model.pt"
            best_checkpoint.write_bytes(b"placeholder")
            config = load_config(Path(__file__).resolve().parents[1] / "configs/expert3_moescore.json")
            config.update({"dataset_root": str(root), "m2d_checkpoint": str(base_checkpoint),
                           "cache_dir": str(root / "cache"), "output_dir": str(root / "runs"),
                           "device": "cpu", "amp": False})
            stub_model = StubModel()
            with patch("inference_expert3.MoescoreExpert3", return_value=stub_model), \
                    patch("inference_expert3.load_best_checkpoint", return_value={"epoch": 2}):
                result = run_inference(config, best_checkpoint, dataset="custom",
                                       metadata_list=metadata, wav_dir=audio_dir)
            self.assertEqual(stub_model.asserted_captions, ["a single caption"])
            self.assertIsNone(result["metrics"])
            with Path(result["prediction_csv"]).open(newline="", encoding="utf-8") as stream:
                rows = list(__import__("csv").DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["wav_file_name"], "one.wav")
            self.assertAlmostEqual(float(rows[0]["pred_score"]), 6.0)

    def test_dataloader_keeps_physical_batch_eight_and_drops_remainder(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.train_expert3 import build_dataloaders

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_split(root, "train", 9)
            self._write_split(root, "validation", 2)
            self._write_split(root, "test", 2)
            cfg = {
                "dataset_root": str(root), "train_csv": "meta_data/train_average.csv",
                "validation_csv": "meta_data/validation_average.csv", "test_csv": "meta_data/test_average.csv",
                "wav_dir": "wav", "filename_column": "wav_file_name", "caption_column": "text",
                "score_column": "average_score", "sample_rate": 16000, "max_len": 10,
                "seed": 42,
            }
            train_loader, val_loader, _ = build_dataloaders(cfg)
            self.assertEqual(train_loader.batch_size, 8)
            self.assertTrue(train_loader.drop_last)
            self.assertEqual(len(train_loader), 1)
            self.assertEqual(next(iter(train_loader))["scores"].numel(), 8)
            self.assertEqual(val_loader.batch_size, 1)

            overlap_audio = np.zeros(1600, dtype=np.float32)
            sf.write(root / "wav/test/validation_0.wav", overlap_audio, 16000)
            with (root / "meta_data/test_average.csv").open("w", newline="", encoding="utf-8") as stream:
                import csv
                writer = csv.writer(stream)
                writer.writerow(("wav_file_name", "text", "average_score"))
                writer.writerow(("validation_0.wav", "overlapping split", 5.0))
                writer.writerow(("test_1.wav", "test sample", 5.0))
            with self.assertRaisesRegex(ValueError, "validation/test split overlap"):
                build_dataloaders(cfg)

    def test_config_rejects_microbatch_accumulation_or_other_batch_size(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.train_expert3 import _check_resume_config, load_config, validate_config

        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/expert3_moescore.json")
        validate_config(cfg)
        _check_resume_config(cfg, dict(cfg))
        for key, changed in (("early_stop_patience", 3), ("train_csv", "other.csv"),
                             ("expected_counts", {"train": 4}),
                             ("split_manifest", {"train": {"metadata_sha256": "changed"}})):
            modified = dict(cfg)
            modified[key] = changed
            with self.subTest(resume_key=key), self.assertRaisesRegex(ValueError, key):
                _check_resume_config(cfg, modified)
        cfg["gradient_accumulation_steps"] = 4
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            validate_config(cfg)

    def test_optimizer_contains_every_and_only_trainable_parameter(self):
        from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.train_expert3 import build_optimizer

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.trainable = nn.Linear(3, 2)
                self.frozen = nn.Linear(2, 1)
                self.frozen.requires_grad_(False)

        model = Tiny()
        optimizer = build_optimizer(model, {"lr": 1e-5, "optimizer": {"name": "Adam", "betas": [0.9, 0.999]}})
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
