from .dataset_generator import DatasetGenerator
import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset generator.")
    p.add_argument(
        "--datasets",
        "-d",
        type=str,
        help="Specify the txt file containing the datasets to generate.",
    )
    p.add_argument(
        "--out", type=str, help="Specify output directory other than default."
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.datasets and not args.out:
        parser.print_help(sys.stderr)
        sys.exit(0)
    else:
        DatasetGenerator.generate(args.datasets, args.out)


if __name__ == "__main__":
    main()
