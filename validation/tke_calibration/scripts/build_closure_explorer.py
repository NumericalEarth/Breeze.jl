"""Build a portable, dependency-free HTML explorer from exported coefficient values."""
import argparse
import csv
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("parameters", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
models = {}
with args.parameters.open() as stream:
    for row in csv.DictReader(stream):
        model = models.setdefault(row["model"], {
            "label": row["model"], "space": row["parameter_space"],
            "protocol": row["protocol"], "status": row["status"],
            "source": row["source"], "parameters": {}})
        model["parameters"][row["parameter"]] = float(row["value"])
if not models:
    parser.error("The parameter file contains no models")
for model in models.values():
    required = {"Cˢ"}
    if model["space"] == "constant":
        required.update(("Cᵘ", "Cᶜ", "Cᵉ", "Cᴰ"))
    elif model["space"] == "ri":
        required.update(key + suffix for key in ("Cᵘ", "Cᶜ", "Cᵉ", "Cᴰ") for suffix in ("⁻", "⁰", "⁺"))
        required.update(("Ri⁰", "Riᵟ"))
    else:
        parser.error(f"Unknown parameter space: {model['space']}")
    if required - model["parameters"].keys():
        parser.error(f"Missing parameters for {model['label']}: {sorted(required - model['parameters'].keys())}")
    parameters = model["parameters"]
    if any(not math.isfinite(v) or v < 0 for v in parameters.values()) or any(parameters[k] <= 0 for k in required - {"Ri⁰"}):
        parser.error(f"Parameters must be finite and nonnegative, with positive transport, dissipation and length scales: {model['label']}")
template = Path(__file__).with_name("closure_explorer_template.html").read_text()
data = json.dumps(list(models.values()), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(template.replace("__MODEL_DATA__", data))
args.output.with_suffix(".json").write_text(data)
print(args.output)
