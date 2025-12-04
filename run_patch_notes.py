from pathlib import Path
from patch_parser import PatchNoteParser

parser = PatchNoteParser()
text = Path("sample_patch_notes.txt").read_text(encoding="utf-8")
for change in parser.parse_patch_note("custom", text):
    ability = change.ability or "(unknown)"
    print(
        f"{change.agent:<10} "
        f"{ability:<15} "
        f"{change.direction.value:<7} "
        f"{change.magnitude.value:<9} "
        f"{change.description}"
    )
