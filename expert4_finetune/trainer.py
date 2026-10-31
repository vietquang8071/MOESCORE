"""Training utilities for Expert 4."""
import os
import random
import tempfile
import csv
import json
import logging
import math
import hashlib
import platform
import subprocess
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def linear_schedule(optimizer, total_steps: int, warmup_steps: int):
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError('Schedule needs positive steps and 0 <= warmup < total')
    def factor(step):
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(path):
    # Resumable experiment checkpoints include Python and NumPy RNG states.
    # Only load checkpoints from a trusted experiment directory.
    return torch.load(path, map_location='cpu', weights_only=False)


def rng_state(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def correlation_metrics(target, prediction):
    import warnings
    from scipy.stats import spearmanr, pearsonr, kendalltau
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.size < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        warnings.warn('Correlation is undefined for fewer than two or constant values', RuntimeWarning)
        return {'srcc': float('nan'), 'lcc': float('nan'), 'ktau': float('nan'),
                'mse': float(np.mean((prediction - target) ** 2))}
    return {'srcc': float(spearmanr(target, prediction).statistic),
            'lcc': float(pearsonr(target, prediction).statistic),
            'ktau': float(kendalltau(target, prediction).statistic),
            'mse': float(np.mean((prediction - target) ** 2))}


def evaluate(model, loader, device):
    from .losses import expert4_loss
    model.eval()
    predictions, targets, rows, loss_sum, count = [], [], [], 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch['wavs'], batch['wav_lens'] = batch['wavs'].to(device), batch['wav_lens'].to(device)
            scores, _ = model(batch)
            loss, _, _ = expert4_loss(scores, batch['scores'].to(device))
            count += len(scores)
            loss_sum += loss.item() * len(scores)
            raw = (scores.cpu() * 5 + 5).tolist()
            predictions.extend(raw)
            targets.extend(batch['scores_raw'].tolist())
            for i, value in enumerate(raw):
                rows.append({'sample_id': batch['sample_ids'][i], 'model_id': batch['model_ids'][i],
                             'model_name': batch['model_names'][i], 'score_raw': batch['scores_raw'][i].item(),
                             'prediction_normalized': scores[i].item(), 'prediction_raw': value})
    return loss_sum / count, correlation_metrics(targets, predictions), rows


def assert_training_gradients(model):
    modules = {'BEATs final layer': model.audio_encoder.model.encoder.layers[-1],
               'audio projection': model.audio_encoder.audio_feats_proj,
               'RoBERTa final layer': model.text_encoder.branch.encoder.layer[-1],
               'text projection': model.text_encoder.projection,
               'audio-to-text attention': model.seq_coattn.audio_to_text_attn,
               'text-to-audio attention': model.seq_coattn.text_to_audio_attn,
               'head': model.head}
    if model.audio_encoder.layer_weights.grad is None:
        raise RuntimeError('BEATs layer aggregation weights received no gradient')
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        if not grads or not any(g is not None for g in grads) or any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f'{name} did not receive finite gradients')
    for module in (model.audio_encoder.model.encoder.layers[0], model.text_encoder.branch.encoder.layer[0]):
        if any(p.grad is not None or p.requires_grad for p in module.parameters()):
            raise RuntimeError('An early frozen encoder layer is trainable or has a gradient')


def trainable_state_dict(model):
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def load_trainable_state_dict(model, state):
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(state) != names:
        raise ValueError(f'Trainable checkpoint mismatch; missing={sorted(names - set(state))}, extra={sorted(set(state) - names)}')
    model.load_state_dict(state, strict=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(source_dir):
    source = Path(source_dir)
    beats = source if (source / 'BEATs.py').is_file() else source / 'beats'
    files = [beats / name for name in ('BEATs.py', 'backbone.py', 'modules.py')]
    if any(path.exists() for path in files) and not all(path.is_file() for path in files):
        raise FileNotFoundError(f'Incomplete BEATs source in {beats}')
    digest = hashlib.sha256()
    if all(path.is_file() for path in files):
        for path in files:
            digest.update(path.name.encode())
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
        source_hash = digest.hexdigest()
    else:
        source_hash = None
    try:
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'git_commit': commit, 'sha256': source_hash}


def _environment(resources, fingerprints):
    import shutil
    commit = fingerprints['unilm_source']['git_commit']
    versions = {}
    for name in ('torch', 'torchaudio', 'transformers', 'laion-clap', 'scipy', 'numpy', 'PyYAML'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'gpu_total_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            'ffmpeg': shutil.which('ffmpeg'), 'packages': versions,
            'unilm_commit': commit, 'fingerprints': fingerprints}


def _logger(output):
    logger = logging.getLogger('expert4_repro')
    logger.setLevel(logging.INFO)
    for previous in list(logger.handlers):
        logger.removeHandler(previous)
        previous.close()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(output / 'train.log', mode='a')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_training(config_path, resume=None, smoke_only=False):
    """Train or resume from the last completed epoch; validation never updates weights."""
    import yaml
    from .config import load_config
    from .dataset import Expert4Dataset, collate_expert4
    from .losses import expert4_loss
    from .model import Expert4

    cfg = load_config(config_path)
    r, t = cfg['resources'], cfg['training']
    output = Path(r['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and not smoke_only and (output / 'last_expert4.pt').exists():
        raise ValueError(f'{output} already has a last checkpoint; set --resume or choose a new OUTPUT_DIR')
    logger = _logger(output)
    if resume is None and not smoke_only:
        with (output / 'training_config.yaml').open('w') as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
    seed = int(t['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    train_ds = Expert4Dataset(r['train_csv'], r['train_audio_dir'])
    val_ds = Expert4Dataset(r['validation_csv'], r['validation_audio_dir'])
    if set(item['sample_id'] for item in train_ds.rows) & set(item['sample_id'] for item in val_ds.rows):
        raise ValueError('Training and validation audio identifiers overlap')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, generator=generator,
                              num_workers=t['num_workers'], collate_fn=collate_expert4)
    val_loader = DataLoader(val_ds, batch_size=t['validation_batch_size'], shuffle=False,
                            num_workers=t['num_workers'], collate_fn=collate_expert4)
    fingerprints = {name: _sha256(r[name]) for name in ('beats_checkpoint', 'clap_checkpoint', 'train_csv', 'validation_csv')}
    fingerprints['unilm_source'] = _source_identity(r['unilm_source_dir'])
    if resume is None and not smoke_only:
        with (output / 'environment.json').open('w') as handle:
            json.dump(_environment(r, fingerprints), handle, indent=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Expert4.from_pretrained(r['beats_checkpoint'], r['clap_checkpoint'], r['unilm_source_dir']).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Parameters: total=%d trainable=%d frozen=%d', total, trainable, total - trainable)
    groups = {'BEATs': 0, 'aggregation': 0, 'audio_projection': 0, 'RoBERTa': 0,
              'text_projection': 0, 'SeqCoAttn': 0, 'head': 0, 'other': 0}
    prefixes = [('audio_encoder.model.', 'BEATs'), ('audio_encoder.layer_weights', 'aggregation'),
                ('audio_encoder.audio_feats_proj.', 'audio_projection'), ('text_encoder.branch.', 'RoBERTa'),
                ('text_encoder.projection.', 'text_projection'), ('seq_coattn.', 'SeqCoAttn'), ('head.', 'head')]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = next((label for prefix, label in prefixes if name.startswith(prefix)), 'other')
            groups[group] += parameter.numel()
    for name, count_in_group in groups.items():
        logger.info('Trainable %s: %d', name, count_in_group)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=t['learning_rate'], betas=tuple(t['adam_betas']))
    total_steps = len(train_loader) * t['epochs']
    warmup_steps = int(total_steps * t['warmup_ratio'])
    scheduler = linear_schedule(optimizer, total_steps, warmup_steps)
    use_amp = bool(t['amp']) and device.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, global_step, best_srcc, stale = 0, 0, -math.inf, 0
    if resume is not None:
        state = load_checkpoint(resume)
        if state['config']['training'] != cfg['training'] or state['fingerprints'] != fingerprints:
            raise ValueError('Resume config or checkpoint/CSV fingerprints differ from the saved run')
        load_trainable_state_dict(model, state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        if state['scaler_state_dict'] is not None:
            scaler.load_state_dict(state['scaler_state_dict'])
        restore_rng(state['rng_state'], generator)
        start_epoch, global_step = state['epoch'] + 1, state['global_step']
        best_srcc, stale = state['best_srcc'], state['stale_epochs']
        logger.info('Resumed at epoch %d, step %d', start_epoch, global_step)
    metrics_path = output / 'metrics.csv'
    fields = ['epoch', 'train_total_loss', 'train_clipped_mse', 'train_contrastive',
              'validation_total_loss', 'validation_srcc', 'validation_lcc',
              'validation_ktau', 'validation_mse', 'learning_rate']
    if start_epoch == 0 and not smoke_only:
        with metrics_path.open('w', newline='') as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    for epoch in range(start_epoch, t['epochs']):
        model.train()
        sums = np.zeros(3, dtype=np.float64)
        count = 0
        for batch in train_loader:
            batch['wavs'] = batch['wavs'].to(device)
            batch['wav_lens'] = batch['wav_lens'].to(device)
            target = batch['scores'].to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    predictions, fused = model(batch)
                    if predictions.shape != target.shape or fused.shape != (len(target), 1024):
                        raise ValueError('Expert 4 output shapes do not match target [B] and fused [B,1024]')
                    loss, mse, contrastive = expert4_loss(predictions.float(), target)
                scaler.scale(loss).backward()
                if global_step == 0 or smoke_only:
                    assert_training_gradients(model)
                if t['gradient_clip_norm'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), t['gradient_clip_norm'])
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if not use_amp or scaler.get_scale() >= scale_before:
                    scheduler.step()
                else:
                    logger.warning('AMP overflow: optimizer step and scheduler update skipped')
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError('CUDA OOM at physical batch size 8. Use a GPU with more memory or enable AMP; batch/accumulation was not silently changed.') from exc
            if not use_amp or scaler.get_scale() >= scale_before:
                global_step += 1
            size = len(target)
            sums += np.array([loss.item(), mse.item(), contrastive.item()]) * size
            count += size
            if smoke_only:
                logger.info('Real-model smoke step succeeded: loss=%f, fused=%s', loss.item(), tuple(fused.shape))
                return
        val_loss, val_metrics, _ = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]['lr']
        record = dict(zip(fields, [epoch + 1, *(sums / count), val_loss,
                                   val_metrics['srcc'], val_metrics['lcc'], val_metrics['ktau'],
                                   val_metrics['mse'], lr]))
        with metrics_path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writerow(record)
        logger.info('Epoch %d: %s', epoch + 1, record)
        improved = math.isfinite(val_metrics['srcc']) and val_metrics['srcc'] > best_srcc
        if improved:
            best_srcc, stale = val_metrics['srcc'], 0
            save_checkpoint(output / 'best_expert4.pt', {'model_state_dict': trainable_state_dict(model),
                'config': cfg, 'fingerprints': fingerprints, 'epoch': epoch, 'best_srcc': best_srcc})
        else:
            stale += 1
            if not math.isfinite(val_metrics['srcc']):
                logger.warning('Validation SRCC undefined; best checkpoint unchanged')
        save_checkpoint(output / 'last_expert4.pt', {
            'model_state_dict': trainable_state_dict(model), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'epoch': epoch, 'global_step': global_step, 'best_srcc': best_srcc,
            'stale_epochs': stale, 'seed': seed, 'rng_state': rng_state(generator),
            'config': cfg, 'fingerprints': fingerprints})
        if stale >= t['early_stop_patience']:
            logger.info('Stopped early after %d epochs without SRCC improvement', stale)
            break
    if not (output / 'best_expert4.pt').is_file():
        raise RuntimeError('No finite validation SRCC was observed; no best checkpoint was selected')
    return output / 'best_expert4.pt'
