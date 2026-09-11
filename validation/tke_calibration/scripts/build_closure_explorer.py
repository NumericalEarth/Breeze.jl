"""Build a portable, dependency-free HTML explorer from exported coefficient values."""
import argparse
import csv
import json
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
template = Path(__file__).with_name("closure_explorer_template.html").read_text()
data = json.dumps(list(models.values()), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(template.replace("__MODEL_DATA__", data))
args.output.with_suffix(".json").write_text(data)
print(args.output)
