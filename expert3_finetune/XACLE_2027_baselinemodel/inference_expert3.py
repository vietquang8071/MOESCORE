"""Inference and labeled-split evaluation for a saved MOESCORE Expert 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.datasets.moescore_expert3_dataset import MoescoreExpert3Dataset
from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import MoescoreExpert3
from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.train_expert3 import compute_metrics, load_config, resolve_path, write_predictions
from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.utils.expert3_checkpoint import load_best_checkpoint


@torch.inference_mode()
def run_inference(config, checkpoint, dataset="validation", dataset_root=None,
                  output=None, batch_size=1, device=None, m2d_checkpoint=None, cache_dir=None,
                  metadata_list=None, wav_dir=None):
    if dataset not in {"validation", "test", "custom"}:
        raise ValueError("dataset must be validation, test, or custom")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dataset == "custom" and (metadata_list is None or wav_dir is None):
        raise ValueError("custom inference requires both metadata_list and wav_dir")
    from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.train_expert3 import prepare_config, _autocast, _move_batch

    cfg = prepare_config(config, dataset_root=dataset_root, checkpoint=m2d_checkpoint,
                         cache_dir=cache_dir, device=device)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for inference but unavailable")
    data_root = Path(cfg["dataset_root"])
    csv_path = Path(metadata_list).expanduser() if metadata_list else data_root / cfg[f"{dataset}_csv"]
    audio_dir = Path(wav_dir).expanduser() if wav_dir else data_root / cfg["wav_dir"] / dataset
    split_dataset = MoescoreExpert3Dataset(
        csv_path, audio_dir,
        sample_rate=cfg["sample_rate"], max_len=cfg["max_len"],
        filename_column=cfg["filename_column"], caption_column=cfg["caption_column"],
        score_column=cfg["score_column"], require_scores=(dataset != "custom"),
    )
    expected = cfg.get("expected_counts", {}).get(dataset) if dataset != "custom" and not metadata_list else None
    if expected is not None and len(split_dataset) != int(expected):
        raise ValueError(f"{dataset}: expected {expected} pairs, found {len(split_dataset)}")
    loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=split_dataset.collate_fn)
    model = MoescoreExpert3(
        checkpoint_path=cfg["m2d_checkpoint"], cache_dir=cfg["cache_dir"],
        activation_checkpointing=cfg["activation_checkpointing"],
        concat_assignment=cfg["concat_assignment"],
        expected_checkpoint_sha256=cfg.get("m2d_checkpoint_sha256"),
    ).to(device)
    checkpoint_info = load_best_checkpoint(model, checkpoint)
    model.eval()
    predictions, targets, filenames = [], [], []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with _autocast(device, bool(cfg["amp"])):
            normalized, _ = model(batch)
        original = normalized.reshape(-1).float().cpu() * 5.0 + 5.0
        predictions.extend(original.tolist())
        if split_dataset.has_scores:
            targets.extend(batch["scores_raw"].reshape(-1).float().cpu().tolist())
        filenames.extend(batch["wav_file_names"])
    metrics = compute_metrics(predictions, targets) if split_dataset.has_scores and len(split_dataset) >= 2 else None

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    result_csv = Path(output).expanduser() if output else checkpoint_path.parent / f"evaluation_result_{dataset}.csv"
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(result_csv, filenames, predictions)
    result_json = result_csv.with_suffix(".json")
    result = {
        "dataset": dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_info["epoch"],
        "metrics": metrics,
        "prediction_csv": str(result_csv.resolve()),
    }
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Run config.json or configs/expert3_moescore.json")
    parser.add_argument("--checkpoint", required=True, help="best_model.pt produced by train_expert3.py")
    parser.add_argument("--dataset", choices=("validation", "test", "custom"), default="validation")
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-list", help="Override the selected split CSV; for custom pairs provide --wav-dir")
    parser.add_argument("--wav-dir", help="Override WAV folder; required with --dataset custom")
    parser.add_argument("--m2d-checkpoint")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    result = run_inference(
        config, args.checkpoint, dataset=args.dataset, dataset_root=args.data_root,
        output=args.output, batch_size=args.batch_size, device=args.device,
        m2d_checkpoint=args.m2d_checkpoint, cache_dir=args.cache_dir,
        metadata_list=args.metadata_list, wav_dir=args.wav_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
