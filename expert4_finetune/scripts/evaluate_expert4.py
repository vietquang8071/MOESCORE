"""Evaluate a validation-selected Expert 4 checkpoint on labeled validation data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert4_finetune.inference import run_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output')
    parser.add_argument('--group-by-model-id', action='store_true')
    args = parser.parse_args()
    _, metrics, path = run_prediction(args.checkpoint, args.config, 'validation', args.output, args.group_by_model_id)
    print(json.dumps({'predictions': str(path), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
