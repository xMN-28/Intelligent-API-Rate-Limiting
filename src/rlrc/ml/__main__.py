"""Unified optional ML command namespace: rlrc ml / python -m rlrc.ml."""

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: rlrc ml {train,hardware,predictor,sim,fidelity,real} [options]")
        print("Use 'rlrc ml <command> --help' for training, hardware probes, or benchmarks.")
        return
    command = sys.argv[1]
    if command in {"train", "hardware"}:
        from rlrc.ml.training import main as training

        training()
    elif command == "predictor":
        import argparse
        from pathlib import Path

        from rlrc.ml.prediction import train_predictor

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--episodes", type=int, default=100)
        args = parser.parse_args(sys.argv[2:])
        print(train_predictor(args.output, args.episodes))
    else:
        from rlrc.ml.benchmarks import main as benchmarks

        benchmarks()


if __name__ == "__main__":
    main()
