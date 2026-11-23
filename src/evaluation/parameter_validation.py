"""Basic validation for parameter-extraction output."""

REQUIRED_KEYS = {
    "module",
    "transcript",
    "meeting_duration",
    "safety_topics",
    "safety_instructions",
    "questions",
    "participation_cues",
    "computer_vision",
    "ppe_assessment",
    "transcript_statistics",
    "evidence_limitations",
    "safety_keywords",
    "language_clarity",
    "additional_signals",
}


def validate_parameter_output(data: dict) -> list[str]:
    """
    Return a list of validation errors.
    An empty list means the basic structure is valid.
    """
    errors = []

    if not isinstance(data, dict):
        return ["Output must be a dictionary."]

    missing_keys = sorted(REQUIRED_KEYS - data.keys())

    if missing_keys:
        errors.append(
            "Missing required keys: " + ", ".join(missing_keys)
        )

    if data.get("module") != "parameter_extraction":
        errors.append("Unexpected module name.")

    duration = data.get("meeting_duration")

    if isinstance(duration, dict):
        seconds = duration.get("seconds")

        if seconds is not None and (
            not isinstance(seconds, (int, float)) or seconds < 0
        ):
            errors.append("Meeting duration must be non-negative or null.")

    topics = data.get("safety_topics")

    if isinstance(topics, dict):
        for topic_name, topic_data in topics.items():
            if not isinstance(topic_data, dict):
                errors.append(
                    f"Safety topic '{topic_name}' must be a dictionary."
                )
                continue

            if not isinstance(topic_data.get("mentioned"), bool):
                errors.append(
                    f"Safety topic '{topic_name}' needs a boolean 'mentioned'."
                )

    instructions = data.get("safety_instructions")

    if isinstance(instructions, dict):
        if not isinstance(instructions.get("items"), list):
            errors.append("Safety instruction items must be a list.")

        if not isinstance(instructions.get("count"), int):
            errors.append("Safety instruction count must be an integer.")
        elif isinstance(instructions.get("items"), list):
            if instructions["count"] != len(instructions["items"]):
                errors.append(
                    "Safety instruction count does not match its items."
                )

    questions = data.get("questions")

    if isinstance(questions, dict):
        if not isinstance(questions.get("items"), list):
            errors.append("Question items must be a list.")

        if not isinstance(questions.get("count"), int):
            errors.append("Question count must be an integer.")
        elif isinstance(questions.get("items"), list):
            if questions["count"] != len(questions["items"]):
                errors.append("Question count does not match its items.")

    limitations = data.get("evidence_limitations")

    if not isinstance(limitations, list):
        errors.append("Evidence limitations must be a list.")

    return errors