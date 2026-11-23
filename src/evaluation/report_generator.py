
import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Build an evidence-based Markdown report for a toolbox talk."""

    def __init__(self):
        self.topic_names = {
            "hazard_awareness": "Hazard awareness",
            "ppe_discussion": "PPE safety discussion",
            "emergency_information": "Emergency information",
            "injury_prevention": "Injury prevention",
            "stop_work_authority": "Stop-work authority",
            "equipment_safety": "Equipment safety",
        }

    @staticmethod
    def _read_json(path: str | Path) -> dict:
        """Read a JSON file and ensure its top-level value is an object."""
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object in {path}, not a list or scalar."
            )

        return data

    @staticmethod
    def _text(value: Any, default: str = "Not available") -> str:
        """Convert a value to safe, readable report text."""
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()
            return value if value else default

        return str(value)

    @staticmethod
    def _list(value: Any) -> list:
        """Return a list, or an empty list for unsupported values."""
        return value if isinstance(value, list) else []

    @staticmethod
    def _get_transcript(data: dict) -> str:
        """Support transcript values stored as text or as a dictionary."""
        transcript = data.get("transcript", "")

        if isinstance(transcript, str):
            return transcript.strip()

        if isinstance(transcript, dict):
            for key in ("text", "transcript", "full_text"):
                value = transcript.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    def _topic_section(self, parameters: dict) -> tuple[str, list, list]:
        topics = parameters.get("safety_topics", {})
        lines = [
            "## 2. Safety topic findings",
            "",
            "| Topic | Finding | Matched keywords |",
            "|---|---|---|",
        ]

        detected = []
        unknown = []

        if not isinstance(topics, dict):
            topics = {}

        for key, name in self.topic_names.items():
            result = topics.get(key)

            if not isinstance(result, dict):
                status = "Unknown"
                keywords = []
                unknown.append(name)
            elif result.get("mentioned") is True:
                status = "Evidence detected"
                keywords = self._list(result.get("matched_keywords"))
                detected.append(name)
            else:
                # A false keyword match does not prove the topic was absent.
                status = "Unknown — no keyword match"
                keywords = self._list(result.get("matched_keywords"))
                unknown.append(name)

            keyword_text = ", ".join(
                str(word) for word in keywords
            ) or "None"

            lines.append(
                f"| {name} | {status} | {keyword_text} |"
            )

        lines.extend([
            "",
            (
                "**Interpretation:** A detected keyword indicates a mention, "
                "not proof that the subject was explained completely or "
                "understood by the workers. No match is treated as unknown."
            ),
            "",
        ])

        return "\n".join(lines), detected, unknown

    @staticmethod
    def _scoring_section(scoring: dict) -> str:
        lines = [
            "## 3. Scoring summary",
            "",
            (
                f"- **Overall quality score:** "
                f"{ReportGenerator._text(scoring.get('overall_score'))}"
            ),
            (
                f"- **Score status:** "
                f"{ReportGenerator._text(scoring.get('score_label'))}"
            ),
            (
                f"- **Topic detection coverage:** "
                f"{ReportGenerator._text(scoring.get('topic_detection_coverage_percentage'))}%"
            ),
        ]

        counts = scoring.get("topic_detection_count")
        if isinstance(counts, dict):
            lines.append(
                f"- **Topics detected:** "
                f"{ReportGenerator._text(counts.get('detected'))} of "
                f"{ReportGenerator._text(counts.get('total_expected'))}"
            )

        lines.extend([
            "",
            (
                "Topic detection coverage is not an overall quality rating. "
                "An overall score is reported only if the scoring engine "
                "provides one; this report does not calculate a replacement."
            ),
            "",
        ])

        return "\n".join(lines)

    @staticmethod
    def _dimension_section(scoring: dict) -> str:
        dimensions = scoring.get("dimensions", {})
        lines = [
            "## 4. Ten-dimension assessment",
            "",
            "| Dimension | Status | Score | Evidence / explanation |",
            "|---|---|---|---|",
        ]

        if not isinstance(dimensions, dict) or not dimensions:
            return (
                "\n".join(lines)
                + "\n\nNo dimension results were available in the scoring file.\n"
            )

        for key, item in dimensions.items():
            if not isinstance(item, dict):
                continue

            name = ReportGenerator._text(item.get("name"), key)
            status = ReportGenerator._text(item.get("status"))
            score = item.get("score")
            max_score = item.get("max_score")

            if score is None:
                score_text = "Not scored"
            elif max_score is not None:
                score_text = f"{score}/{max_score}"
            else:
                score_text = str(score)

            evidence = ReportGenerator._text(item.get("evidence"))
            evidence = evidence.replace("\n", " ").replace("|", "\\|")

            lines.append(
                f"| {name} | {status} | {score_text} | {evidence} |"
            )

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _computer_vision_section(parameters: dict) -> str:
        cv = parameters.get("computer_vision", {})
        ppe = parameters.get("ppe_assessment", {})

        lines = [
            "## 5. Computer vision and PPE observations",
            "",
        ]

        if not isinstance(cv, dict):
            cv = {}

        if not isinstance(ppe, dict):
            ppe = {}

        duration = parameters.get("meeting_duration")
        if isinstance(duration, dict):
            duration = duration.get(
                "duration_seconds",
                duration.get("seconds"),
            )

        if duration is None:
            lines.append("- **Meeting duration:** Not available")
        else:
            lines.append(
                f"- **Meeting duration (seconds):** {duration}"
            )

        for key, label in (
            ("frames_analyzed", "Frames analyzed"),
            ("frames_with_people", "Frames with detected people"),
            ("frames_with_visible_faces", "Frames with visible faces"),
            (
                "frames_with_possible_group_gathering",
                "Frames with possible group gathering",
            ),
        ):
            value = cv.get(key)
            if value is not None:
                lines.append(f"- **{label}:** {value}")

        predictions = ppe.get("predictions")
        if isinstance(predictions, dict):
            lines.extend([
                "",
                "### PPE model predictions",
                "",
                "| Label | Prediction count |",
                "|---|---:|",
            ])

            for label, count in predictions.items():
                lines.append(f"| {label} | {count} |")

        compliance = ppe.get("compliance_percentage")
        lines.append("")
        if compliance is None:
            lines.append(
                "- **PPE compliance percentage:** Not calculated"
            )
        else:
            lines.append(
                f"- **PPE compliance percentage:** {compliance}%"
            )

        limitations = []
        limitations.extend(ReportGenerator._list(cv.get("limitations")))
        limitations.extend(ReportGenerator._list(
            parameters.get("evidence_limitations")
        ))

        ppe_limitation = ppe.get("limitation")
        if isinstance(ppe_limitation, str) and ppe_limitation:
            limitations.append(ppe_limitation)

        if limitations:
            lines.extend(["", "### Visual-analysis limitations", ""])
            for item in dict.fromkeys(limitations):
                lines.append(f"- {item}")

        lines.extend([
            "",
            (
                "Visual predictions require human verification. Repeated "
                "detections across frames do not establish unique attendance, "
                "attention, understanding, or PPE compliance."
            ),
            "",
        ])

        return "\n".join(lines)

    @staticmethod
    def _transcript_section(parameters: dict) -> str:
        transcript = ReportGenerator._get_transcript(parameters)
        stats = parameters.get("transcript_statistics", {})

        lines = [
            "## 6. Transcript and communication evidence",
            "",
        ]

        if transcript:
            lines.extend([
                "### Transcript",
                "",
                transcript,
                "",
            ])
        else:
            lines.extend([
                "No transcript text was available in the parameter file.",
                "",
            ])

        if isinstance(stats, dict) and stats:
            lines.append("### Transcript statistics")
            lines.append("")

            for key, label in (
                ("word_count", "Word count"),
                ("sentence_count", "Sentence count"),
                ("average_words_per_sentence", "Average words per sentence"),
                ("timestamped_segment_count", "Timestamped segments"),
            ):
                value = stats.get(key)
                if value is not None:
                    lines.append(f"- **{label}:** {value}")

            clarity = stats.get("clarity_assessment")
            if clarity:
                lines.append(
                    f"- **Clarity assessment:** {clarity}"
                )

            lines.append("")

        lines.append(
            "Transcript text may contain speech-recognition errors or omissions."
        )
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _strengths_and_review(
        detected: list,
        unknown: list,
        parameters: dict,
    ) -> str:
        lines = [
            "## 7. Evidence-supported strengths",
            "",
        ]

        if detected:
            for topic in detected:
                lines.append(
                    f"- The transcript contains keyword evidence related "
                    f"to **{topic}**."
                )
        else:
            lines.append(
                "- No safety topic had a confirmed keyword match in the "
                "available parameter results."
            )

        lines.extend([
            "",
            "## 8. Areas for review",
            "",
        ])

        if unknown:
            for topic in unknown:
                lines.append(
                    f"- **{topic}:** Review the original recording and "
                    "transcript to determine whether this subject was covered."
                )
        else:
            lines.append(
                "- No unmatched topics were reported, but human review "
                "is still recommended."
            )

        lines.extend([
            "- Verify PPE detections against the original video.",
            "- Review the transcript for recognition errors.",
            "- Use validated criteria before assigning an overall quality score.",
            (
                "- Assess worker engagement, attention, and understanding "
                "using appropriate evidence; do not infer them from "
                "face visibility or head direction alone."
            ),
            "",
        ])

        return "\n".join(lines)

    def generate_report(
        self,
        parameters: dict,
        scoring: dict,
    ) -> str:
        """Return a complete Markdown evaluation report."""
        if not isinstance(parameters, dict):
            raise ValueError("Parameter extraction data must be a dictionary.")

        if not isinstance(scoring, dict):
            raise ValueError("Scoring results must be a dictionary.")

        topic_text, detected, unknown = self._topic_section(parameters)

        sections = [
            "# Toolbox Talk Evaluation Report",
            "",
            "> This report summarizes automated observations. It is not a "
            "certification of workplace safety or worker understanding.",
            "",
            "## 1. Executive summary",
            "",
            (
                f"- **Safety topics with keyword evidence:** "
                f"{len(detected)} of {len(self.topic_names)} configured topics."
            ),
            (
                f"- **Topics requiring review:** {len(unknown)}."
            ),
            (
                f"- **Overall quality score:** "
                f"{self._text(scoring.get('overall_score'))}."
            ),
            "",
            topic_text,
            self._scoring_section(scoring),
            self._dimension_section(scoring),
            self._computer_vision_section(parameters),
            self._transcript_section(parameters),
            self._strengths_and_review(detected, unknown, parameters),
        ]

        limitations = []
        limitations.extend(self._list(parameters.get("evidence_limitations")))
        limitations.extend(self._list(parameters.get("limitations")))
        limitations.extend(self._list(scoring.get("limitations")))

        sections.extend([
            "## 9. Limitations and responsible use",
            "",
        ])

        if limitations:
            for limitation in dict.fromkeys(limitations):
                sections.append(f"- {limitation}")
        else:
            sections.append(
                "- The available inputs did not provide a limitations list."
            )

        sections.extend([
            "",
            (
                "This report is intended to support human review. Automated "
                "keyword matches and model predictions can be incomplete "
                "or incorrect. Verify important findings against the source "
                "recording and established safety procedures."
            ),
            "",
        ])

        return "\n".join(sections)

    def generate_from_files(
        self,
        parameter_path: str | Path,
        scoring_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Read existing JSON inputs and save the Markdown report."""
        parameters = self._read_json(parameter_path)
        scoring = self._read_json(scoring_path)

        report = self.generate_report(parameters, scoring)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")

        return output


def main():
    generator = ReportGenerator()

    output = generator.generate_from_files(
        parameter_path="data/output/parameter_extraction.json",
        scoring_path="data/output/scoring_results.json",
        output_path="data/output/final_evaluation_report.md",
    )

    print("\nFINAL EVALUATION REPORT")
    print("=======================")
    print("Report generated successfully.")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
