import argparse
import csv
import json
import runpy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from patch_parser import PatchNoteParser
from sklearn.metrics import classification_report

# Default to the new 11.08 Agent Updates annotations.
DEFAULT_DATA_PATH = Path("data/test_11-08_data.csv")
DEFAULT_HTML_PATH = Path("public/patch-notes-html/valorant-patch-notes-11-08.html")


def _coerce_float(value: Any) -> Optional[float]:
    """Convert raw CSV values to floats when possible."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "unknoiown"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _canonical_text(
    text: str, agent: Optional[str] = None, ability: Optional[str] = None, strip_ability: bool = True
) -> str:
    """Normalize description to align parsed output with manual annotations."""
    cleaned = " ".join(text.split()).strip()
    lowered = cleaned.lower()

    def strip_prefix(prefix: Optional[str]) -> None:
        nonlocal cleaned, lowered
        if not prefix:
            return
        prefix_clean = prefix.strip().lower()
        if prefix_clean and lowered.startswith(prefix_clean + " "):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()

    strip_prefix(agent)
    if strip_ability:
        strip_prefix(ability)
    return cleaned.lower()


def load_jsonl_annotations(path: Path) -> List[Dict[str, Any]]:
    """Read annotated text entries from a JSON Lines file."""
    annotations: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            annotations.append(json.loads(line))
    return annotations


def load_csv_annotations(path: Path) -> List[Dict[str, Any]]:
    """Load annotations from the manual CSV (Agent Updates numeric-only for 11.08)."""
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("description") or "").strip()
            direction = (row.get("direction") or "").strip().lower()
            magnitude = (row.get("magnitude") or "").strip().lower()

            entry: Dict[str, Any] = {
                "text": text,
                "direction": direction,
                "magnitude": magnitude,
                "old_value": _coerce_float(row.get("old_value")),
                "new_value": _coerce_float(row.get("new_value")),
                "meta": {
                    "lineNum": row.get("lineNum"),
                    "agent": row.get("agent"),
                    "ability": row.get("ability"),
                    "unit": row.get("unit"),
                    "patch_version": row.get("patch_version"),
                },
            }
            entries.append(entry)
    return entries


def load_py_dataframe(path: Path) -> List[Dict[str, Any]]:
    """Load annotations from a .py file containing a pandas DataFrame named `df`."""
    context = runpy.run_path(str(path))
    df = context.get("df")
    if df is None:
        raise ValueError(f"No DataFrame named `df` found in {path}")
    entries: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        entries.append(
            {
                "text": str(row.get("description", "")).strip(),
                "direction": str(row.get("direction", "")).strip().lower(),
                "magnitude": str(row.get("magnitude", "")).strip().lower(),
                "old_value": _coerce_float(row.get("old_value")),
                "new_value": _coerce_float(row.get("new_value")),
                "meta": {
                    "lineNum": row.get("lineNum"),
                    "agent": row.get("agent"),
                    "ability": row.get("ability"),
                    "unit": row.get("unit"),
                    "patch_version": row.get("patch_version"),
                },
            }
        )
    return entries


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    """Load annotations regardless of whether they come from JSONL, CSV, or .py DataFrame."""
    if path.suffix == ".jsonl":
        return load_jsonl_annotations(path)
    if path.suffix == ".csv":
        return load_csv_annotations(path)
    if path.suffix == ".py":
        return load_py_dataframe(path)
    raise ValueError(f"Unsupported annotation format for {path}")


def parse_patch(path: Path) -> List[Dict[str, Any]]:
    """Parse an HTML patch note and return structured changes as dicts."""
    parser = PatchNoteParser()
    html = path.read_text(encoding="utf-8")
    changes = parser.parse_patch_html(html, fallback_version="11.08")
    df = parser.to_dataframe(changes)
    return df.to_dict(orient="records")


def build_lookup(
    parsed_changes: List[Dict[str, Any]],
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Index parsed rows by (agent, ability, canonical_desc), by (agent, canonical_desc), and by canonical_desc alone."""
    by_ability: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    by_agent: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_text: Dict[str, Dict[str, Any]] = {}
    for row in parsed_changes:
        agent = (row.get("agent") or "").strip().lower()
        ability = (row.get("ability") or "").strip().lower()
        desc = row.get("description", "")
        canonical_no_ability = _canonical_text(desc, row.get("agent"), row.get("ability"), strip_ability=True)
        canonical_with_ability = _canonical_text(desc, row.get("agent"), row.get("ability"), strip_ability=False)

        if agent and canonical_no_ability:
            by_agent[(agent, canonical_no_ability)] = row
            if ability:
                by_ability[(agent, ability, canonical_no_ability)] = row

        for canon in {canonical_no_ability, canonical_with_ability}:
            if canon and canon not in by_text:
                by_text[canon] = row
    return by_ability, by_agent, by_text


def evaluate(
    entries: Iterable[Dict[str, Any]],
    parsed_lookup: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]] = None,
    parsed_lookup_agent: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    parsed_lookup_text: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_unknown: bool = False,
) -> None:
    parser = PatchNoteParser()

    y_true_dir: List[str] = []
    y_pred_dir: List[str] = []
    y_true_mag: List[str] = []
    y_pred_mag: List[str] = []
    mismatches: List[Dict[str, Any]] = []
    missing_from_parsed: List[Dict[str, Any]] = []

    for entry in entries:
        gold_dir = entry["direction"]
        gold_mag = entry["magnitude"]
        if skip_unknown and (gold_dir == "unknown" or gold_mag == "unknown"):
            continue

        text = entry["text"]
        agent = (entry.get("meta", {}).get("agent") or "").strip().lower()
        ability = (entry.get("meta", {}).get("ability") or "").strip().lower()
        canonical = _canonical_text(text, agent, ability)

        pred_dir_val: Optional[str] = None
        pred_mag_val: Optional[str] = None

        if parsed_lookup is not None:
            row = parsed_lookup.get((agent, ability, canonical))
            if row is None and parsed_lookup_agent is not None:
                row = parsed_lookup_agent.get((agent, canonical))
            if row is None and parsed_lookup_text is not None:
                row = parsed_lookup_text.get(canonical)
            if row:
                pred_dir_val = str(row.get("direction", "")).strip().lower()
                pred_mag_val = str(row.get("magnitude", "")).strip().lower()
            else:
                missing_from_parsed.append(entry)
                continue

        if pred_dir_val is None or pred_mag_val is None:
            old_val = entry.get("old_value")
            new_val = entry.get("new_value")
            pred_dir = parser.detect_direction(text)
            pred_mag = parser.estimate_magnitude(text, old_val, new_val)
            pred_dir_val = pred_dir.value
            pred_mag_val = pred_mag.value

        y_true_dir.append(gold_dir)
        y_pred_dir.append(pred_dir_val)
        y_true_mag.append(gold_mag)
        y_pred_mag.append(pred_mag_val)

        if gold_dir != pred_dir_val or gold_mag != pred_mag_val:
            mismatches.append(
                {
                    "text": text,
                    "gold_direction": gold_dir,
                    "pred_direction": pred_dir_val,
                    "gold_magnitude": gold_mag,
                    "pred_magnitude": pred_mag_val,
                    "meta": entry.get("meta", {}),
                }
            )

    if not y_true_dir:
        print("[WARN] No entries evaluated (possibly all skipped or unmatched).")
        return

    dir_labels = sorted(set(y_true_dir + y_pred_dir))
    mag_labels = sorted(set(y_true_mag + y_pred_mag))

    print("=== Direction Classification ===")
    print(
        classification_report(
            y_true_dir, y_pred_dir, labels=dir_labels, target_names=dir_labels, zero_division=0
        )
    )
    print("=== Magnitude Estimation ===")
    print(
        classification_report(
            y_true_mag, y_pred_mag, labels=mag_labels, target_names=mag_labels, zero_division=0
        )
    )

    if parsed_lookup is not None:
        print(f"Matched parsed rows: {len(y_true_dir)} / {len(entries)}")
        if missing_from_parsed:
            print(f"Unmatched annotations (not found in parsed output): {len(missing_from_parsed)}")
            for miss in missing_from_parsed:
                meta = miss.get("meta", {})
                print(
                    f"- {miss['text']} (agent={meta.get('agent')}, ability={meta.get('ability')}, line={meta.get('lineNum')})"
                )

    if mismatches:
        print("\n=== Misclassified Examples ===")
        for mismatch in mismatches:
            meta = mismatch.get("meta") or {}
            meta_bits = [f"{k}={v}" for k, v in meta.items() if v not in (None, "", "None")]
            meta_str = f" ({', '.join(meta_bits)})" if meta_bits else ""
            print(f"- text: {mismatch['text']}{meta_str}")
            print(
                f"  direction: gold={mismatch['gold_direction']} pred={mismatch['pred_direction']}"
            )
            print(
                f"  magnitude: gold={mismatch['gold_magnitude']} pred={mismatch['pred_magnitude']}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate direction & magnitude heuristics against manual annotations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to annotations (.csv for 11.08 Agent Updates, .jsonl legacy, or .py DataFrame).",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=DEFAULT_HTML_PATH,
        help="HTML patch note to parse (defaults to 11.08). Use --no-html to fall back to heuristics only.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip parsing HTML and evaluate direct text heuristics only.",
    )
    parser.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Skip rows with unknown direction/magnitude so metrics focus on labeled items.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Annotated data not found at {args.data}")

    entries = load_annotations(args.data)
    parsed_lookup = None
    parsed_lookup_agent = None
    parsed_lookup_text = None

    if not args.no_html:
        if not args.html.exists():
            raise FileNotFoundError(f"Patch HTML not found at {args.html}")
        parsed_changes = parse_patch(args.html)
        parsed_lookup, parsed_lookup_agent, parsed_lookup_text = build_lookup(parsed_changes)

    evaluate(
        entries,
        parsed_lookup=parsed_lookup,
        parsed_lookup_agent=parsed_lookup_agent,
        parsed_lookup_text=parsed_lookup_text,
        skip_unknown=args.skip_unknown,
    )


if __name__ == "__main__":
    main()
