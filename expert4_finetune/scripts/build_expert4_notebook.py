"""Package the standalone Expert 4 source into a fresh-runtime Colab notebook."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / 'expert4_repro').glob('*.py')) + [
    ROOT / 'scripts' / name for name in ('train_expert4.py', 'evaluate_expert4.py', 'predict_expert4.py', 'check_beats.py', 'check_clap.py')
] + [ROOT / 'configs' / 'expert4_colab.yaml']


def cell(kind, source, number):
    result = {'cell_type': kind, 'id': f'expert4-{number}', 'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        result.update({'execution_count': None, 'outputs': []})
    return result


with io.BytesIO() as stream:
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FILES:
            archive.write(path, path.relative_to(ROOT).as_posix())
    encoded = base64.b64encode(stream.getvalue()).decode('ascii')

cells = [
cell('markdown', '''# Fine-tune MOESCORE Expert 4 on Colab

Run cells from top to bottom. Edit the Drive path cell before running preflight. The notebook carries a snapshot of `expert4_repro/` and the three CLI commands; no unpublished repository branch is required. `train_and_predict` trains, validates, evaluates, and predicts test scores. `predict_only` loads a prior best checkpoint and predicts test scores. Neither mode uses test data for optimization or checkpoint selection.

The output CSV uses generic columns including `wav_file_name` and `prediction_raw`. Official 2027 submission formatting has not been verified.
''', 0),
cell('code', '''import sys, importlib.util, shutil, platform, subprocess, json
print('Python:', platform.python_version())
if sys.version_info >= (3, 13):
    raise RuntimeError('LAION-CLAP needs NumPy <2, which has no Python 3.13 wheel. In Colab choose Runtime > Change runtime type > Runtime Version 2026.07 (Python 3.12), then rerun from the first cell.')
torch_probe = ('import json, torch; print(json.dumps({' 
    '"torch": torch.__version__, "cuda": torch.version.cuda, '
    '"available": torch.cuda.is_available(), '
    '"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, '
    '"memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None}))')
runtime = json.loads(subprocess.check_output([sys.executable, '-c', torch_probe], text=True))
print('PyTorch:', runtime['torch'], 'CUDA runtime:', runtime['cuda'])
print('CUDA available:', runtime['available'])
if not runtime['available']:
    raise RuntimeError('Select a GPU runtime in Colab before continuing.')
print('GPU:', runtime['gpu'])
print('GPU memory GiB:', runtime['memory_gib'])
for name in ('torchaudio', 'torchcodec', 'transformers', 'laion_clap', 'scipy', 'yaml'):
    print(name, 'installed:', importlib.util.find_spec(name) is not None)
print('FFmpeg:', shutil.which('ffmpeg'))
''', 1),
cell('code', '''import subprocess
from pathlib import Path
torch_version = runtime['torch'].split('+')[0]
constraint = Path('/content/expert4_torch_constraint.txt')
torchaudio_probe = 'import torchaudio, torchaudio.compliance.kaldi; print(torchaudio.__version__)'
audio_check = subprocess.run([sys.executable, '-c', torchaudio_probe], capture_output=True, text=True)
if audio_check.returncode:
    cuda = runtime['cuda']
    if not cuda:
        raise RuntimeError('PyTorch has no CUDA runtime; choose a GPU Colab runtime')
    index = 'https://download.pytorch.org/whl/cu' + cuda.replace('.', '')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', f'torchaudio=={torch_version}', '--index-url', index])
    audio_check = subprocess.run([sys.executable, '-c', torchaudio_probe], capture_output=True, text=True)
if audio_check.returncode:
    raise RuntimeError(f'TorchAudio still cannot import: {audio_check.stderr}')
torchaudio_version = audio_check.stdout.strip().splitlines()[-1]
if torchaudio_version.split('+')[0] != torch_version:
    raise RuntimeError(f'TorchAudio {torchaudio_version} must match PyTorch {runtime["torch"]}.')
constraint.write_text(f'torch=={torch_version}\\ntorchaudio=={torch_version}\\n')
requirements = ['numpy==1.26.4', 'laion-clap==1.1.7', 'transformers>=4.51.3,<5', 'scipy>=1.15.0', 'PyYAML', 'librosa', 'torchlibrosa', 'SoundFile', 'wget', 'ftfy', 'tqdm']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--only-binary=numpy', '--upgrade-strategy', 'only-if-needed', '-c', str(constraint), *requirements])
dependency_probe = ('import numpy, torch, torchaudio, laion_clap, transformers; '
                    'print("NumPy", numpy.__version__, "PyTorch", torch.__version__, '
                    '"TorchAudio", torchaudio.__version__, "Transformers", transformers.__version__, "LAION-CLAP import: OK")')
try:
    subprocess.check_call([sys.executable, '-c', dependency_probe])
except subprocess.CalledProcessError as exc:
    raise RuntimeError('Dependencies failed to import in a fresh Python process. Restart the Colab session once after installation and rerun from the first cell; if the same error persists, inspect the package traceback above.') from exc
print('TorchCodec and FFmpeg are not used for the SciPy WAV dataset loader.')
''', 2),
cell('code', '''from google.colab import drive
drive.mount('/content/drive')
''', 3),
cell('code', '''# Edit these values. None triggers bounded discovery for input files when practical.
MODE = 'train_and_predict'  # or 'predict_only'
DRIVE_ROOT = '/content/drive/MyDrive'
DATASET_ROOT = None
TRAIN_CSV = None
VALIDATION_CSV = None
TEST_CSV = None
TRAIN_AUDIO_DIR = None
VALIDATION_AUDIO_DIR = None
TEST_AUDIO_DIR = None
BEATS_CHECKPOINT = None
CLAP_CHECKPOINT = None
UNILM_SOURCE_DIR = None  # existing Git checkout, or clone official UniLM below
OUTPUT_DIR = None  # REQUIRED: choose a Drive directory for this run
BEST_CHECKPOINT = None  # optional; inferred from OUTPUT_DIR/best_expert4.pt
RESUME_CHECKPOINT = None  # optional; OUTPUT_DIR/last_expert4.pt to resume
EPOCHS = 50
WARMUP_RATIO = 0.1  # implementation assumption
EARLY_STOP_PATIENCE = 10  # implementation assumption
AMP = True
GRADIENT_CLIP_NORM = None
''', 4),
cell('code', f'''import base64, io, zipfile, sys
from pathlib import Path
PROJECT_ROOT = Path('/content/moescore_expert4_repro')
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE = '{encoded}'
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
    archive.extractall(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print('Expert 4 source ready:', PROJECT_ROOT)
''', 5),
cell('code', '''import os, subprocess, json
from pathlib import Path
if MODE not in ('train_and_predict', 'predict_only'):
    raise ValueError('MODE must be train_and_predict or predict_only')
if OUTPUT_DIR is None:
    raise ValueError('Set OUTPUT_DIR to a Google Drive directory before continuing')
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
if UNILM_SOURCE_DIR is None:
    unilm = Path('/content/unilm_expert4')
    if not (unilm / 'beats' / 'BEATs.py').is_file():
        subprocess.check_call(['git', 'clone', '--filter=blob:none', '--sparse', 'https://github.com/microsoft/unilm.git', str(unilm)])
        subprocess.check_call(['git', '-C', str(unilm), 'sparse-checkout', 'set', 'beats'])
else:
    unilm = Path(UNILM_SOURCE_DIR)
if not (unilm / 'beats' / 'BEATs.py').is_file() and not (unilm / 'BEATs.py').is_file():
    raise FileNotFoundError(f'No official BEATs.py found under {unilm}')
try:
    unilm_commit = subprocess.check_output(['git', '-C', str(unilm), 'rev-parse', 'HEAD'], text=True).strip()
except subprocess.CalledProcessError as exc:
    raise RuntimeError('UniLM source must be a Git checkout so its commit can be recorded') from exc
saved_environment = output_dir / 'environment.json'
if (RESUME_CHECKPOINT or MODE == 'predict_only') and saved_environment.is_file():
    expected_commit = json.loads(saved_environment.read_text()).get('unilm_commit')
    if expected_commit and unilm_commit != expected_commit:
        if UNILM_SOURCE_DIR is not None:
            raise RuntimeError(f'Existing UniLM checkout is {unilm_commit}; this run needs {expected_commit}')
        subprocess.check_call(['git', '-C', str(unilm), 'fetch', 'origin', expected_commit])
        subprocess.check_call(['git', '-C', str(unilm), 'checkout', expected_commit])
        unilm_commit = subprocess.check_output(['git', '-C', str(unilm), 'rev-parse', 'HEAD'], text=True).strip()
print('UniLM source:', unilm, 'commit:', unilm_commit)
''', 6),
cell('code', '''import csv
from pathlib import Path
def discover_one(root, names, required=True, max_depth=8, max_dirs=12000):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    matches, scanned = [], 0
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        dirs[:] = [d for d in dirs if not d.startswith('.') and depth < max_depth]
        scanned += 1
        if scanned > max_dirs:
            raise RuntimeError(f'Discovery scanned {max_dirs} directories under {root}; set the path explicitly')
        matches.extend(Path(current) / name for name in files if name in names)
    if len(matches) > 1:
        raise ValueError('Several candidates found; set the path explicitly: ' + str(matches[:20]))
    if not matches and required:
        raise FileNotFoundError(f'No {names} under {root}; set the path explicitly')
    return matches[0] if matches else None
search_root = Path(DATASET_ROOT or DRIVE_ROOT)
def path_or_find(explicit, names, root=search_root, required=True):
    path = Path(explicit) if explicit else discover_one(root, names, required=required)
    if path is not None and not path.is_file():
        raise FileNotFoundError(path)
    return path
train_csv = path_or_find(TRAIN_CSV, {'train_average.csv'}) if MODE == 'train_and_predict' else (Path(TRAIN_CSV) if TRAIN_CSV else None)
validation_csv = path_or_find(VALIDATION_CSV, {'validation_average.csv'}) if MODE == 'train_and_predict' else (Path(VALIDATION_CSV) if VALIDATION_CSV else None)
test_csv = path_or_find(TEST_CSV, {'test_average.csv', 'test.csv'}, required=True)
def audio_dir(explicit, csv_path, split):
    path = Path(explicit) if explicit else (csv_path.parent.parent / 'wav' / split if csv_path else None)
    if path is not None and not path.is_dir():
        raise FileNotFoundError(f'Set {split} audio directory explicitly: {path}')
    return path
train_audio_dir = audio_dir(TRAIN_AUDIO_DIR, train_csv, 'train')
validation_audio_dir = audio_dir(VALIDATION_AUDIO_DIR, validation_csv, 'validation')
test_audio_dir = audio_dir(TEST_AUDIO_DIR, test_csv, 'test')
beats_checkpoint = path_or_find(BEATS_CHECKPOINT, {'BEATs_iter3_plus_AS2M.pt'}, root=Path(DRIVE_ROOT))
clap_checkpoint = path_or_find(CLAP_CHECKPOINT, {'630k-audioset-best.pt', 'laion-clap.pt'}, root=Path(DRIVE_ROOT))
best_checkpoint = Path(BEST_CHECKPOINT) if BEST_CHECKPOINT else output_dir / 'best_expert4.pt'
if MODE == 'predict_only' and not best_checkpoint.is_file():
    raise FileNotFoundError(f'Set BEST_CHECKPOINT to a trained Expert 4 checkpoint: {best_checkpoint}')
for split, path in [('train', train_csv), ('validation', validation_csv), ('test', test_csv)]:
    if path:
        with path.open(newline='', encoding='utf-8-sig') as handle:
            headers = next(csv.reader(handle))
        required = {'wav_file_name', 'text'}
        if not required.issubset(set(headers)):
            raise ValueError(f'{split} headers {headers} do not contain {required}')
        if split != 'test' and 'average_score' not in headers:
            raise ValueError(f'{split} lacks average_score: {headers}')
        print(split, path, 'headers:', headers)
print('Paths: train', train_csv, train_audio_dir, 'validation', validation_csv, validation_audio_dir)
print('Paths: test', test_csv, test_audio_dir, 'BEATs', beats_checkpoint, 'CLAP', clap_checkpoint, 'output', output_dir)
''', 7),
cell('code', '''import yaml
from expert4_repro.config import DEFAULTS, load_config
from copy import deepcopy
config_path = output_dir / 'training_config.yaml'
cfg = load_config(config_path, validate_paths=False) if MODE == 'predict_only' and config_path.is_file() else deepcopy(DEFAULTS)
cfg['resources'].update({'train_csv': str(train_csv) if train_csv else None,
    'validation_csv': str(validation_csv) if validation_csv else None,
    'test_csv': str(test_csv), 'train_audio_dir': str(train_audio_dir) if train_audio_dir else None,
    'validation_audio_dir': str(validation_audio_dir) if validation_audio_dir else None,
    'test_audio_dir': str(test_audio_dir), 'beats_checkpoint': str(beats_checkpoint),
    'clap_checkpoint': str(clap_checkpoint), 'unilm_source_dir': str(unilm), 'output_dir': str(output_dir)})
if MODE == 'train_and_predict':
    cfg['training'].update({'epochs': EPOCHS, 'warmup_ratio': WARMUP_RATIO,
                            'early_stop_patience': EARLY_STOP_PATIENCE, 'amp': AMP,
                            'gradient_clip_norm': GRADIENT_CLIP_NORM})
with config_path.open('w') as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
load_config(config_path, validate_paths=MODE == 'train_and_predict')
print('Resolved training configuration:', config_path)
''', 8),
cell('code', '''if MODE == 'train_and_predict':
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts/check_beats.py'),
        '--checkpoint', str(beats_checkpoint), '--unilm-source-dir', str(unilm)])
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts/check_clap.py'),
        '--checkpoint', str(clap_checkpoint)])
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts/train_expert4.py'),
        '--config', str(config_path), '--smoke-test'])
    print('Real BEATs/CLAP checkpoint, aggregation, masks, forward/backward and optimizer checks completed')
''', 9),
cell('code', '''if MODE == 'train_and_predict':
    command = [sys.executable, str(PROJECT_ROOT / 'scripts/train_expert4.py'), '--config', str(config_path)]
    if RESUME_CHECKPOINT:
        command += ['--resume', str(RESUME_CHECKPOINT)]
    subprocess.check_call(command)
    best_checkpoint = output_dir / 'best_expert4.pt'
print('Best checkpoint:', best_checkpoint)
''', 10),
cell('code', '''if validation_csv is not None and validation_audio_dir is not None:
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts/evaluate_expert4.py'),
        '--config', str(config_path), '--checkpoint', str(best_checkpoint),
        '--output', str(output_dir / 'validation_predictions.csv'), '--group-by-model-id'])
''', 11),
cell('code', '''subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts/predict_expert4.py'),
    '--config', str(config_path), '--checkpoint', str(best_checkpoint),
    '--output', str(output_dir / 'test_predictions.csv')])
print('Test predictions:', output_dir / 'test_predictions.csv')
''', 12),
cell('markdown', '''Resume after a disconnection: remount Drive, set the same resource paths and `OUTPUT_DIR`, set `RESUME_CHECKPOINT` to `OUTPUT_DIR/last_expert4.pt`, keep `MODE='train_and_predict'`, and run all cells. The training command resumes after the last completed epoch. A CUDA out-of-memory error at batch 8 stops training; do not treat gradient accumulation as equivalent for the pairwise loss.
''', 13),
]

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
           'language_info': {'name': 'python'}, 'colab': {'name': 'Finetune_MOESCORE_Expert4.ipynb'}},
           'nbformat': 4, 'nbformat_minor': 5}
destination = ROOT / 'colab' / 'Finetune_MOESCORE_Expert4.ipynb'
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
print(destination, 'bytes', destination.stat().st_size)
