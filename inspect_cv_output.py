
import json

with open(
    "data/output/parameter_extraction.json",
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)

cv = data.get("computer_vision", {})

print("\n=== COMPUTER VISION OUTPUT ===")
print("Top-level keys:", list(cv.keys()))

for key, value in cv.items():
    print(f"\n--- {key} ---")
    if isinstance(value, dict):
        print("Nested keys:", list(value.keys()))
        print(json.dumps(value, indent=2, ensure_ascii=False)[:3000])
    else:
        print(str(value)[:1000])

