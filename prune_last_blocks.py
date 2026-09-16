"""Prune the final two ViT blocks of a first-stage MedCore checkpoint."""

from medcore_pruning.utils import parse_config
from prune import build_parser, run_pruning


def main(argv=None):
    args = parse_config(build_parser(last_blocks=True), argv)
    run_pruning(args, last_blocks=True)


if __name__ == "__main__":
    main()
