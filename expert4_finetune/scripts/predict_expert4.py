"""Predict XACLE test scores in CSV order; no test labels are needed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert4_finetune.inference import run_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    _, _, path = run_prediction(args.checkpoint, args.config, 'test', args.output)
    print(f'Test predictions: {path}')


if __name__ == '__main__':
    main()
