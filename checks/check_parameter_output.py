
import json
from pathlib import Path

from src.evaluation.parameter_validation import validate_parameter_output

output_path = Path("data/output/parameter_extraction.json")

with output_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

sections = [
    "safety_topics",
    "safety_instructions",
    "safety_keywords",
    "language_clarity",
    "additional_signals",
    "ppe_assessment",
]

print("\n========== PARAMETER EXTRACTION AUDIT ==========\n")

for section in sections:
    print(f"\n--- {section} ---")

    if section in data:
        print(json.dumps(data[section], indent=2, ensure_ascii=False))
    else:
        print("MISSING: This section is not in the saved JSON.")

errors = validate_parameter_output(data)

print("\n========== VALIDATION ==========")
print("Result:", "PASS" if not errors else "FAIL")
print("Errors:", errors)
