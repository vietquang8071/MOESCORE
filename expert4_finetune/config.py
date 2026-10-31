"""Configuration validation for the fixed Expert 4 reproduction decisions."""
from copy import deepcopy
from pathlib import Path
import yaml

DEFAULTS = {
    'resources': {'train_csv': None, 'validation_csv': None, 'test_csv': None,
                  'train_audio_dir': None, 'validation_audio_dir': None, 'test_audio_dir': None,
                  'beats_checkpoint': None, 'clap_checkpoint': None,
                  'unilm_source_dir': None, 'output_dir': None},
    'training': {'batch_size': 8, 'validation_batch_size': 8, 'epochs': 50,
                 'learning_rate': 1e-5, 'adam_betas': [0.9, 0.999],
                 'warmup_ratio': 0.1, 'seed': 42, 'num_workers': 0,
                 'early_stop_patience': 10, 'amp': True,
                 'gradient_clip_norm': None},
    'loss': {'tau': 0.25, 'margin': 0.1, 'beta': 1.0, 'gamma': 0.5},
}


def load_config(path, validate_paths=True):
    with Path(path).open(encoding='utf-8') as handle:
        values = yaml.safe_load(handle) or {}
    cfg = deepcopy(DEFAULTS)
    for section, mapping in values.items():
        if section not in cfg or not isinstance(mapping, dict):
            raise ValueError(f'Unknown or malformed configuration section: {section}')
        unknown = set(mapping) - set(cfg[section])
        if unknown:
            raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')
        cfg[section].update(mapping)
    train, loss = cfg['training'], cfg['loss']
    fixed = {'batch_size': 8, 'learning_rate': 1e-5, 'adam_betas': [0.9, 0.999]}
    for key, expected in fixed.items():
        if train[key] != expected:
            raise ValueError(f'{key} must be {expected} for this reproduction')
    for key, expected in {'tau': 0.25, 'margin': 0.1, 'beta': 1.0, 'gamma': 0.5}.items():
        if loss[key] != expected:
            raise ValueError(f'loss.{key} must be {expected}')
    if train['epochs'] < 1 or train['validation_batch_size'] < 1 or train['num_workers'] < 0 or not 0 <= train['warmup_ratio'] < 1 or train['early_stop_patience'] < 1:
        raise ValueError('Invalid training schedule or DataLoader setting')
    if validate_paths:
        required = ['train_csv', 'validation_csv', 'train_audio_dir', 'validation_audio_dir',
                    'beats_checkpoint', 'clap_checkpoint', 'unilm_source_dir', 'output_dir']
        missing = [name for name in required if not cfg['resources'][name]]
        if missing:
            raise ValueError(f'Configure these resource paths: {missing}')
        for name in required:
            if name != 'output_dir' and not Path(cfg['resources'][name]).exists():
                raise FileNotFoundError(f'{name}: {cfg["resources"][name]}')
    return cfg
