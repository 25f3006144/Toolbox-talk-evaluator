
"""Evidence-aware scoring for the Toolbox Talk Evaluator."""

import json
from pathlib import Path
from typing import Any


class ScoringEngine:
    """Evaluate available evidence without inventing missing scores."""

    DIMENSIONS = {
        "safety_topic_coverage": "Safety topic coverage",
        "hazard_awareness": "Hazard awareness",
        "ppe_safety": "PPE safety",
        "emergency_preparedness": "Emergency preparedness",
        "injury_prevention": "Injury prevention",
        "stop_work_authority": "Stop-work authority",
        "equipment_safety": "Equipment safety",
        "communication_clarity": "Communication clarity",
        "worker_engagement": "Worker engagement",
        "demonstration_visual_support": "Demonstration and visual support",
    }

    TOPIC_MAP = {
        "hazard_awareness": "hazard_awareness",
        "ppe_safety": "ppe_discussion",
        "emergency_preparedness": "emergency_information",
        "injury_prevention": "injury_prevention",
        "stop_work_authority": "stop_work_authority",
        "equipment_safety": "equipment_safety",
    }

    def score(self, parameter_data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameter_data, dict):
            raise TypeError("parameter_data must be a dictionary.")

        topics = parameter_data.get("safety_topics")
        if not isinstance(topics, dict):
            raise ValueError(
                "parameter_data must contain a safety_topics dictionary."
            )

        dimensions: dict[str, dict[str, Any]] = {}
        valid_topic_results = []

        # Score positive topic detections as evidence indicators.
        # A missing keyword match is not proof of absence.
        for dimension, topic_key in self.TOPIC_MAP.items():
            topic = topics.get(topic_key)

            if (
                not isinstance(topic, dict)
                or not isinstance(topic.get("mentioned"), bool)
            ):
                dimensions[dimension] = self._unknown(
                    self.DIMENSIONS[dimension],
                    "Reliable topic evidence is unavailable.",
                )
                continue

            valid_topic_results.append(topic["mentioned"])

            if topic["mentioned"]:
                keywords = topic.get("matched_keywords", [])
                dimensions[dimension] = {
                    "name": self.DIMENSIONS[dimension],
                    "score": 10,
                    "max_score": 10,
                    "status": "evidence_detected",
                    "evidence_source": "transcript_keyword_detection",
                    "evidence": (
                        "Configured transcript keywords matched. "
                        f"Matched keywords: {keywords}"
                    ),
                }
            else:
                dimensions[dimension] = self._unknown(
                    self.DIMENSIONS[dimension],
                    "No configured keyword matched. This does not prove "
                    "the topic was absent.",
                )

        # Coverage is a separate metric, not part of the overall score.
        valid_count = len(valid_topic_results)
        detected_count = sum(valid_topic_results)

        coverage_percentage = (
            round(detected_count / valid_count * 100, 2)
            if valid_count
            else None
        )

        dimensions["safety_topic_coverage"] = {
            "name": self.DIMENSIONS["safety_topic_coverage"],
            "score": coverage_percentage,
            "max_score": 100,
            "status": "measured" if valid_count else "unknown",
            "evidence_source": "transcript_keyword_detection",
            "evidence": (
                f"{detected_count} of {valid_count} available topic results "
                "had a keyword match."
            ),
            "topics_with_results": valid_count,
            "topics_detected": detected_count,
        }

        # Preserve video predictions without treating them as compliance.
        ppe = parameter_data.get("ppe_assessment", {})
        if not isinstance(ppe, dict):
            ppe = {}

        dimensions["ppe_safety"]["video_evidence"] = {
            "available": ppe.get("video_predictions_available", False),
            "predictions": ppe.get("predictions", {}),
            "compliance_percentage": ppe.get("compliance_percentage"),
            "limitation": ppe.get(
                "limitation",
                "Video predictions do not establish individual compliance.",
            ),
        }

        # These dimensions need better validated evidence.
        for dimension in (
            "communication_clarity",
            "worker_engagement",
            "demonstration_visual_support",
        ):
            dimensions[dimension] = self._unknown(
                self.DIMENSIONS[dimension],
                "The current pipeline does not provide sufficient "
                "validated evidence to score this dimension.",
            )

        # No overall quality score is claimed until scoring rules for
        # positive and negative evidence are validated for all dimensions.
        evidence_detected = sum(
            result.get("status") == "evidence_detected"
            for key, result in dimensions.items()
            if key != "safety_topic_coverage"
        )
        unknown_dimensions = sum(
            result.get("status") == "unknown"
            for key, result in dimensions.items()
            if key != "safety_topic_coverage"
        )

        return {
            "module": "scoring_engine",
            "overall_score": None,
            "score_scale": 100,
            "score_label": (
                "Unavailable: overall quality scoring rules are not "
                "yet validated"
            ),
            "topic_detection_coverage_percentage": coverage_percentage,
            "topic_detection_count": {
                "detected": detected_count,
                "with_results": valid_count,
                "total_expected": len(self.TOPIC_MAP),
            },
            "evidence_summary": {
                "dimensions_with_positive_topic_evidence": evidence_detected,
                "dimensions_unknown": unknown_dimensions,
                "total_dimensions": len(self.DIMENSIONS),
            },
            "dimensions": dimensions,
            "limitations": [
                "Topic keyword matches do not prove understanding or talk quality.",
                "No keyword match is unknown, not proof of topic absence.",
                "Topic coverage is reported separately to avoid double counting.",
                "PPE frame detections are not unique-worker counts.",
                "PPE compliance cannot be calculated from these predictions alone.",
                "An overall quality score is withheld until all scoring rules "
                "are appropriately validated.",
            ],
        }

    @staticmethod
    def _unknown(name: str, reason: str) -> dict[str, Any]:
        return {
            "name": name,
            "score": None,
            "max_score": 10,
            "status": "unknown",
            "evidence": reason,
        }


def main() -> None:
    input_path = Path("data/output/parameter_extraction.json")
    output_path = Path("data/output/scoring_results.json")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Parameter extraction output not found: {input_path}. "
            "Run the parameter extraction pipeline first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        parameter_data = json.load(file)

    results = ScoringEngine().score(parameter_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("SCORING ENGINE RESULTS")
    print("======================")
    print("Overall quality score: Unavailable (rules not yet validated)")
    print(
        "Topic detection coverage:",
        results["topic_detection_coverage_percentage"],
    )
    print(
        "Topics detected:",
        results["topic_detection_count"]["detected"],
        "of",
        results["topic_detection_count"]["with_results"],
        "with results",
    )
    print("\nDIMENSIONS")
    for key, result in results["dimensions"].items():
        score = result.get("score")
        status = result.get("status")
        if key == "safety_topic_coverage":
            print(f"{result['name']}: {score}% ({status})")
        elif status == "evidence_detected":
            print(f"{result['name']}: evidence detected (not a quality score)")
        else:
            print(f"{result['name']}: Unknown")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()


