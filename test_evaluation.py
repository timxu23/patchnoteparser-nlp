from patch_parser import PatchNoteParser, ChangeDirection, Magnitude
from sklearn.metrics import classification_report

# Sample gold-standard data
test_data = [
    {
        "text": "Damage decreased 75 >>> 50",
        "true_direction": ChangeDirection.NERF,
        "true_magnitude": Magnitude.SIGNIFICANT,
    },
    {
        "text": "Duration increased 4.0 >>> 4.5 seconds",
        "true_direction": ChangeDirection.BUFF,
        "true_magnitude": Magnitude.MINOR,
    },
    {
        "text": "VFX on rocket's trail slightly reduced",
        "true_direction": ChangeDirection.NEUTRAL,
        "true_magnitude": Magnitude.MINOR,
    },
]

parser = PatchNoteParser()

y_true_dir = []
y_pred_dir = []
y_true_mag = []
y_pred_mag = []

for entry in test_data:
    text = entry["text"]
    pred_dir = parser.detect_direction(text)
    pred_mag = parser.estimate_magnitude(text)

    y_true_dir.append(entry["true_direction"].value)
    y_pred_dir.append(pred_dir.value)

    y_true_mag.append(entry["true_magnitude"].value)
    y_pred_mag.append(pred_mag.value)

print("=== Direction Classification ===")
print(classification_report(y_true_dir, y_pred_dir))

print("=== Magnitude Estimation ===")
print(classification_report(y_true_mag, y_pred_mag))
