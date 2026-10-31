"""Train or resume standalone MOESCORE Expert 4."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert4_finetune.trainer import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', help='Trusted last_expert4.pt from the same run')
    parser.add_argument('--smoke-test', action='store_true', help='Run one real batch, backward and optimizer step, then exit')
    args = parser.parse_args()
    result = run_training(args.config, resume=args.resume, smoke_only=args.smoke_test)
    if result:
        print(f'Best Expert 4 checkpoint: {result}')


if __name__ == '__main__':
    main()
