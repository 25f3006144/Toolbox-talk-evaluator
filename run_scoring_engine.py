
"""Run the Toolbox Talk scoring engine."""

import json
from pathlib import Path

from src.evaluation.scoring_engine import ScoringEngine


def main():
    input_path = Path("data/output/parameter_extraction.json")
    output_path = Path("data/output/scoring_results.json")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run the parameter extraction pipeline first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        parameter_data = json.load(file)

    results = ScoringEngine().score(parameter_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nTOOLBOX TALK SCORING RESULTS")
    print("=" * 35)

    print("Overall score:", results["overall_score"], "/ 100")
    print(
        "Overall score status:",
        results["score_label"],
    )

    print(
        "\nTopic detection coverage:",
        results["topic_detection_coverage_percentage"],
        "%",
    )

    counts = results["topic_detection_count"]
    print(
        "Topics detected:",
        counts["detected"],
        "of",
        counts["with_results"],
    )

    print("\nDIMENSION RESULTS")
    print("-" * 35)

    for key, dimension in results["dimensions"].items():
        score = dimension.get("score")
        status = dimension.get("status", "not scored")

        if key == "safety_topic_coverage":
            print(f"{dimension['name']}: {score}% ({status})")
        elif status == "evidence_detected":
            print(f"{dimension['name']}: Evidence detected")
        else:
            print(f"{dimension['name']}: Unknown")

    print("\nEvidence summary:")
    summary = results["evidence_summary"]
    print(
        "Dimensions with positive topic evidence:",
        summary["dimensions_with_positive_topic_evidence"],
    )
    print("Dimensions with unknown results:", summary["dimensions_unknown"])

    print("\nLimitations:")
    for limitation in results["limitations"]:
        print("-", limitation)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

