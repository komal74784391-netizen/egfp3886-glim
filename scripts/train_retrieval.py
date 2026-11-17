"""Train the GLIM tabular retrieval model."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kdd24.yaml")
    args = parser.parse_args()
    print("Training GLIM tabular retrieval with", args.config)


if __name__ == "__main__":
    main()
