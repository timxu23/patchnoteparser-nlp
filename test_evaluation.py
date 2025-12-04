import argparse
import json
from pathlib import Path

from patch_parser import PatchNoteParser
from sklearn.metrics import classification_report

DEFAULT_DATA_PATH = Path("data/annotated_changes.jsonl")


def load_annotations(path: Path):
    """Read annotated text entries from a JSON Lines file."""
    annotations = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            annotations.append(json.loads(line))
    return annotations


def evaluate(entries):
    parser = PatchNoteParser()

    y_true_dir = []
    y_pred_dir = []
    y_true_mag = []
    y_pred_mag = []
    mismatches = []

    for entry in entries:
        text = entry["text"]
        pred_dir = parser.detect_direction(text)
        pred_mag = parser.estimate_magnitude(text)

        gold_dir = entry["direction"]
        gold_mag = entry["magnitude"]

        y_true_dir.append(gold_dir)
        y_pred_dir.append(pred_dir.value)

        y_true_mag.append(gold_mag)
        y_pred_mag.append(pred_mag.value)

        if gold_dir != pred_dir.value or gold_mag != pred_mag.value:
            mismatches.append(
                {
                    "text": text,
                    "gold_direction": gold_dir,
                    "pred_direction": pred_dir.value,
                    "gold_magnitude": gold_mag,
                    "pred_magnitude": pred_mag.value,
                }
            )

    print("=== Direction Classification ===")
    print(classification_report(y_true_dir, y_pred_dir, zero_division=0))
    print("=== Magnitude Estimation ===")
    print(classification_report(y_true_mag, y_pred_mag, zero_division=0))

    if mismatches:
        print("\n=== Misclassified Examples ===")
        for mismatch in mismatches:
            print(f"- text: {mismatch['text']}")
            print(
                f"  direction: gold={mismatch['gold_direction']} pred={mismatch['pred_direction']}"
            )
            print(
                f"  magnitude: gold={mismatch['gold_magnitude']} pred={mismatch['pred_magnitude']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate patch note heuristics.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the annotated JSON Lines dataset.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Annotated data not found at {args.data}")

    entries = load_annotations(args.data)
    evaluate(entries)


if __name__ == "__main__":
    main()
