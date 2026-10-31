"""Independent Expert 4 validation and test prediction export."""
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import Expert4Dataset, collate_expert4
from .model import Expert4
from .trainer import _sha256, _source_identity, correlation_metrics, load_checkpoint, load_trainable_state_dict


def predict_rows(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            batch['wavs'] = batch['wavs'].to(device)
            batch['wav_lens'] = batch['wav_lens'].to(device)
            predictions, fused = model(batch)
            if predictions.shape != (len(batch['captions']),) or fused.shape != (len(batch['captions']), 1024):
                raise ValueError('Expert 4 prediction/fusion output shape mismatch')
            if not torch.isfinite(predictions).all() or (predictions.abs() > 1.00001).any():
                raise ValueError('Nonfinite or out-of-range normalized predictions')
            for index, value in enumerate(predictions.cpu().tolist()):
                row = {'sample_id': batch['sample_ids'][index], 'wav_file_name': batch['sample_ids'][index],
                       'text': batch['captions'][index], 'model_id': batch['model_ids'][index],
                       'model_name': batch['model_names'][index],
                       'prediction_normalized': value, 'prediction_raw': value * 5 + 5}
                if batch['scores_raw'] is not None:
                    row['score_raw'] = batch['scores_raw'][index].item()
                rows.append(row)
    return rows


def export_predictions(rows, path):
    if not rows:
        raise ValueError('No predictions to export')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_finetuned(checkpoint_path, resources, device):
    state = load_checkpoint(checkpoint_path)
    for name in ('beats_checkpoint', 'clap_checkpoint'):
        if _sha256(resources[name]) != state['fingerprints'][name]:
            raise ValueError(f'Pretrained checkpoint identity differs for {name}')
    if _source_identity(resources['unilm_source_dir']) != state['fingerprints']['unilm_source']:
        raise ValueError('UniLM BEATs source differs from the saved training run')
    model = Expert4.from_pretrained(resources['beats_checkpoint'], resources['clap_checkpoint'], resources['unilm_source_dir'])
    load_trainable_state_dict(model, state['model_state_dict'])
    return model.to(device).eval(), state


def run_prediction(checkpoint_path, config_path, split, output_path=None, group_by_model=False):
    from .config import load_config
    cfg = load_config(config_path, validate_paths=False)
    if split not in ('validation', 'test'):
        raise ValueError('Prediction split must be validation or test')
    resources = cfg['resources']
    csv_path, audio_dir = resources[f'{split}_csv'], resources[f'{split}_audio_dir']
    if not csv_path or not audio_dir:
        raise ValueError(f'Configure {split}_csv and {split}_audio_dir')
    dataset = Expert4Dataset(csv_path, audio_dir, require_score=split == 'validation')
    loader = DataLoader(dataset, batch_size=cfg['training']['validation_batch_size'],
                        shuffle=False, num_workers=cfg['training']['num_workers'], collate_fn=collate_expert4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, state = load_finetuned(checkpoint_path, resources, device)
    rows = predict_rows(model, loader, device)
    if output_path is None:
        output_path = Path(resources['output_dir']) / ('validation_predictions.csv' if split == 'validation' else 'test_predictions.csv')
    export_predictions(rows, output_path)
    metrics = None
    if split == 'validation':
        metrics = correlation_metrics([row['score_raw'] for row in rows], [row['prediction_raw'] for row in rows])
        if group_by_model:
            groups = {}
            for row in rows:
                groups.setdefault(str(row['model_id']), []).append(row)
            metrics['by_model_id'] = {name: correlation_metrics([r['score_raw'] for r in group],
                [r['prediction_raw'] for r in group]) for name, group in groups.items()}
    return rows, metrics, Path(output_path)
