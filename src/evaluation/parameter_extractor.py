
import re
from collections import Counter
from typing import Any


class ParameterExtractor:
    """
    Extract structured safety-meeting parameters from:
    1. A speech-to-text transcript
    2. Computer vision results
    3. Optional video/audio metadata

    The output contains evidence-based observations.
    It does not make a final safety score.
    """

    SAFETY_TOPICS = {
       "hazard_awareness": [
            "hazard",
            "hazards",
            "unsafe",
            "unsafe act",
            "unsafe acts",
            "unsafe condition",
            "unsafe conditions",
            "risk",
            "risks",
            "danger",
            "dangerous",
            "potential hazard",
            "look out for",
            "watch out for",
        ],
       "ppe_discussion": [
            "ppe",
            "personal protective equipment",
            "hardhat",
            "hard hat",
            "hard hats",
            "safety helmet",
            "safety helmets",
            "helmet",
            "helmets",
            "safety vest",
            "safety vests",
            "high visibility vest",
            "high visibility clothing",
            "protective eyewear",
            "safety glasses",
            "goggles",
            "hearing protection",
            "ear protection",
            "respirator",
            "respirators",
            "face mask",
            "face masks",
            "protective gloves",
            "safety boots",
            "steel toe boots",
        ],
       "emergency_information": [
            "emergency",
            "emergencies",
            "emergency exit",
            "emergency exits",
            "emergency procedure",
            "emergency procedures",
            "emergency response",
            "emergency contact",
            "emergency number",
            "call 911",
            "call emergency services",
            "evacuate",
            "evacuation",
            "evacuation route",
            "fire alarm",
            "fire extinguisher",
            "first aid",
            "medical assistance",
            "assembly point",
            "muster point",
            "shelter in place",
        ],
        "injury_prevention": [
            "injury prevention",
            "prevent injuries",
            "prevent injury",
            "avoid injury",
            "avoid injuries",
            "get hurt",
            "gets hurt",
            "getting hurt",
            "serious injury",
            "serious injuries",
            "save their life",
            "save your life",
            "protect your life",
            "protect workers",
            "go home safe",
            "go home safely",
            "safety first",
            "look out for each other",
            "take care of yourselves",
            "take care of each other",
        ],
        "stop_work_authority": [
            "stop work",
            "stop the job",
            "stop them",
            "stop an unsafe act",
            "stop unsafe acts",
            "stop unsafe work",
            "right to stop",
            "right to stop work",
            "has the right to stop",
            "authority to stop",
            "stop work authority",
            "unsafe condition",
            "report unsafe",
            "speak up",
        ],
        "equipment_safety": [
            "equipment safety",
            "safe equipment operation",
            "operate safely",
            "operating safely",
            "operating equipment",
            "heavy equipment",
            "moving equipment",
            "equipment moving",
            "machinery",
            "machine safety",
            "vehicle safety",
            "moving vehicles",
            "forklift",
            "forklifts",
            "crane",
            "cranes",
            "power tools",
            "tool safety",
            "equipment inspection",
            "inspect equipment",
            "lockout tagout",
            "lock out tag out",
            "pinch point",
            "pinch points",
        ],
    }

    def extract_parameters(
        self,
        transcript: Any,
        cv_results: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Main function: combines transcript, CV results,
        and optional metadata into one structured dictionary.
        """

        cv_results = cv_results or {}
        metadata = metadata or {}

        transcript_text, segments = self._read_transcript(transcript)
        
        normalized_text = self._normalize(transcript_text)

        topic_results = self._extract_topics(normalized_text)
        instruction_results = self._extract_safety_instructions(transcript_text)
        safety_keyword_results = self._extract_safety_keywords(transcript_text)
        language_clarity_results = self._extract_language_clarity(transcript_text)
        
        
        question_results = self._extract_questions(segments, transcript_text)
       
        language_results = self._extract_transcript_statistics(
            transcript_text, segments
        )
        duration_results = self._extract_duration(
            metadata, segments
        )
        cv_summary = self._extract_cv_summary(cv_results)
        additional_signals = self._extract_additional_signals(
            cv_summary=cv_summary,
            metadata=metadata or {},
        )
        ppe_predictions = cv_summary.get("ppe_predictions") or {}

        ppe_assessment = {
            "speech_mentioned": topic_results["ppe_discussion"]["mentioned"],
            "video_predictions_available": bool(ppe_predictions),
            "predictions": ppe_predictions,
            "compliance_percentage": None,
            "limitation": (
                "PPE detections are predictions across sampled frames, "
                "not verified unique workers. A reliable PPE compliance "
                "percentage cannot be calculated from these counts alone."
            ),
        }        

        participation = {
            "question_like_segments": question_results["count"],
            "visual_group_gathering_cue": cv_summary[
                "possible_group_gathering"
            ],
            "interpretation": (
                "These are limited participation cues. They do not prove "
                "who spoke, who asked a question, or who paid attention."
            ),
        }

        return {
            "module": "parameter_extraction",
            "transcript": {
                "available": bool(transcript_text.strip()),
                "word_count": language_results["word_count"],
                "text": transcript_text,
            },
            "meeting_duration": duration_results,
            "safety_topics": topic_results,
            "safety_instructions": instruction_results,
            "safety_keywords": safety_keyword_results,
            "language_clarity": language_clarity_results,
            "questions": question_results,
            "participation_cues": participation,
            "computer_vision": cv_summary,
            "ppe_assessment": ppe_assessment,  
            "additional_signals": additional_signals,          
            "transcript_statistics": language_results,
            "evidence_limitations": self._build_limitations(
                transcript_text, cv_results, metadata
            ),
        }

    def _read_transcript(self, transcript: Any) -> tuple[str, list]:
        """
        Accept either a plain string or a dictionary containing
        text and optional timestamped segments.
        """

        if isinstance(transcript, str):
            return transcript.strip(), []

        if isinstance(transcript, dict):
            text = transcript.get("transcript", transcript.get("text", ""))
            segments = transcript.get("segments", [])

            if isinstance(text, str) and isinstance(segments, list):
                if not text.strip():
                    text = " ".join(
                        str(segment.get("text", ""))
                        for segment in segments
                        if isinstance(segment, dict)
                    )

                return text.strip(), segments

        return "", []

    def _normalize(self, text: str) -> str:
        """
        Lowercase the text and remove extra whitespace.
        This makes keyword matching more consistent.
        """

        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_topics(self, normalized_text: str) -> dict:
        """
        Search the transcript for each safety topic.
        Return matched keywords and evidence phrases.
        """

        results = {}

        for topic, keywords in self.SAFETY_TOPICS.items():
            matched_keywords = []

            for keyword in keywords:
                pattern = r"\b" + re.escape(keyword) + r"\b"

                if re.search(pattern, normalized_text):
                    matched_keywords.append(keyword)

            results[topic] = {
                "mentioned": bool(matched_keywords),
                "matched_keywords": matched_keywords,
                "evidence_type": "transcript_keyword_match",
            }

        return results
    def _extract_safety_instructions(self, transcript_text: str) -> dict:
        """
        Find simple safety instructions and recommendations in the transcript.
        These are approximate text-based cues, not proof of correct delivery.
        """
        instruction_patterns = [
            (r"\bwear\s+(?:your|a|the)\s+[^.!?]+", "instruction"),
            (r"\buse\s+(?:your|a|the)\s+[^.!?]+", "instruction"),
            (r"\bkeep\s+[^.!?]+", "instruction"),
            (r"\bmake sure\s+(?:you|we|everyone)\s+[^.!?]+", "recommendation"),
            (r"\blook out for\s+[^.!?]+", "recommendation"),
            (r"\bwatch out for\s+[^.!?]+", "recommendation"),
            (r"\bstop work\b[^.!?]*", "instruction"),
            (r"\bbe safe\b", "recommendation"),
            (r"\bspeak up\b[^.!?]*", "recommendation"),
            (r"\breport\s+[^.!?]+", "instruction"),
            (r"\bdo not\s+[^.!?]+", "instruction"),
            (r"\bdon't\s+[^.!?]+", "instruction"),
        ]

        items = []
        seen = set()

        # Search sentence by sentence so we retain a readable evidence phrase.
        sentences = re.split(r"(?<=[.!?])\s+", transcript_text.strip())

        for sentence in sentences:
            for pattern, instruction_type in instruction_patterns:
                for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                    evidence = match.group(0).strip()
                    key = evidence.lower()

                    if key not in seen:
                        seen.add(key)
                        items.append({
                            "text": evidence,
                            "type": instruction_type,
                            "evidence_type": "transcript_text",
                        })

        return {
            "count": len(items),
            "items": items,
            "limitation": (
                "These are approximate text-pattern matches. "
                "They do not prove that an instruction was explained "
                "correctly, understood, or followed."
            ),
        }  

    
    def _extract_safety_keywords(self, transcript_text: str) -> dict:
        """Find safety-related keywords in the transcript."""

        keyword_groups = {
            "hazards": [
                "hazard", "unsafe", "danger", "risk"
            ],
            "ppe": [
                "hardhat", "hard hat", "safety vest",
                "safety glasses", "gloves", "mask", "ppe"
            ],
            "emergency": [
                "emergency", "first aid", "evacuation",
                "emergency exit"
            ],
            "injury_prevention": [
                "injury", "hurt", "accident", "prevent"
            ],
            "stop_work": [
                "stop work", "stop them", "stop the job"
            ],
            "equipment": [
                "equipment", "machinery", "tool", "vehicle"
            ],
        }

        text = transcript_text.lower()
        results = {}

        for category, keywords in keyword_groups.items():
            found = []

            for keyword in keywords:
                pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"

                if re.search(pattern, text):
                    found.append(keyword)

            results[category] = {
                "count": len(found),
                "keywords": found,
            }

        return {
            "categories": results,
            "limitation": (
                "Keyword matches show which words appeared in the "
                "transcript. They do not prove understanding."
            ),
        }
    
  
    def _extract_language_clarity(self, transcript_text: str) -> dict:
        """Calculate basic text indicators, not a definitive clarity score."""

        words = re.findall(r"\b[\w'-]+\b", transcript_text)
        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", transcript_text)
            if sentence.strip()
        ]

        word_count = len(words)
        sentence_count = len(sentences)

        average_words = (
            round(word_count / sentence_count, 2)
            if sentence_count else None
        )

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_words_per_sentence": average_words,
            "transcript_available": bool(transcript_text.strip()),
            "clarity_score": None,
            "limitation": (
                "Sentence length is only a basic text indicator. "
                "It does not measure pronunciation, intelligibility, "
                "grammar quality, or worker understanding."
            ),
        }
    
    def _extract_additional_signals(
        self,
        cv_summary: dict,
        metadata: dict,
    ) -> dict:
        """Expose available evidence without inventing missing measurements."""

        ppe_predictions = cv_summary.get("ppe_predictions") or {}

        return {
            "attendance_percentage": None,
            "attention_level": None,
            "participation_level": None,
            "demonstration_detected": None,
            "environment_cues": {
                "available": (
                    cv_summary.get("module_summaries", {})
                    .get("scene_analysis")
                ),
            },
          
            "talk_duration_seconds": metadata.get("duration_seconds"),
            "time_of_day": metadata.get("recording_time"),
            "supervisor_presence": metadata.get("supervisor_present"),
            "visual_aids_or_boards": metadata.get("visual_aids_present"),
            "ppe_prediction_counts": ppe_predictions,
            "limitations": [
                "Attendance percentage requires a reliable expected group size.",
                "Attention and participation require suitable observable evidence.",
                "Demonstration detection requires a validated visual cue.",
                "Recording time must come from trusted recording metadata.",
                "Supervisor presence and visual aids must not be guessed.",
                "PPE frame detections are not counts of unique workers.",
            ],
        }    




    def _extract_questions(
        self, segments: list, transcript_text: str
    ) -> dict:
        """
        Count transcript segments that look like questions.
        This is a rough text-based signal, not speaker identification.
        """

        question_segments = []

        if segments:
            for segment in segments:
                if not isinstance(segment, dict):
                    continue

                text = str(segment.get("text", "")).strip()

                if self._looks_like_question(text):
                    question_segments.append({
                        "text": text,
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                    })
        else:
            sentences = re.split(
                r"(?<=[.!?])\s+", transcript_text.strip()
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if self._looks_like_question(sentence):
                    question_segments.append({
                        "text": sentence,
                        "start": None,
                        "end": None,
                    })

        return {
            "count": len(question_segments),
            "items": question_segments,
            "evidence_type": "question_like_transcript_text",
            "limitation": (
                "A question-like sentence does not identify its speaker "
                "or prove that a worker participated."
            ),
        }

    def _looks_like_question(self, text: str) -> bool:
        """Detect explicit questions and common question structures."""

        if not text or not text.strip():
            return False

        if "?" in text:
            return True

        normalized = self._normalize(text)

        question_patterns = (
            "who is ", "who are ", "who was ", "who were ",
            "who can ", "who should ", "who will ",
            "what is ", "what are ", "what was ", "what were ",
            "what do ", "what does ", "what did ",
            "what can ", "what should ", "what will ",
            "when is ", "when are ", "when was ", "when were ",
            "when do ", "when does ", "when did ",
            "where is ", "where are ", "where was ", "where were ",
            "where do ", "where does ", "where did ",
            "why is ", "why are ", "why was ", "why were ",
            "why do ", "why does ", "why did ",
            "how is ", "how are ", "how was ", "how were ",
            "how do ", "how does ", "how did ",
            "which is ", "which are ", "which was ", "which were ",
            "which do ", "which does ", "which did ",
        )

        if normalized.startswith(question_patterns):
            return True

        yes_no_openings = (
            "can we ", "can i ",
            "could we ", "could i ",
            "do we ", "does this ",
            "is it ", "is there ",
            "are we ", "are there ",
            "should we ", "would it ",
            "will we ", "did we ",
            "have we ", "has anyone ",
        )

        return normalized.startswith(yes_no_openings)    

    def _extract_transcript_statistics(
        self, transcript_text: str, segments: list
    ) -> dict:
        """
        Calculate basic text statistics.
        These statistics do not determine whether speech is clear.
        """

        words = re.findall(r"\b[\w'-]+\b", transcript_text)
        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", transcript_text)
            if sentence.strip()
        ]

        word_counts = Counter(word.lower() for word in words)

        repeated_words = {
            word: count
            for word, count in word_counts.items()
            if count >= 3 and len(word) > 2
        }

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "average_words_per_sentence": (
                round(len(words) / len(sentences), 2)
                if sentences else 0
            ),
            "repeated_words": repeated_words,
            "timestamped_segment_count": len(segments),
            "clarity_assessment": (
                "Not assessed. Text statistics alone cannot reliably "
                "measure pronunciation, intelligibility, or language clarity."
            ),
        }

    def _extract_duration(
        self, metadata: dict, segments: list
    ) -> dict:
        """
        Prefer a known video/audio duration from metadata.
        Otherwise, estimate a lower bound from transcript timestamps.
        """

        duration = metadata.get("duration_seconds")
        source = "metadata"

        if not isinstance(duration, (int, float)) or duration < 0:
            end_times = [
                segment.get("end")
                for segment in segments
                if isinstance(segment, dict)
                and isinstance(segment.get("end"), (int, float))
            ]

            if end_times:
                duration = max(end_times)
                source = "last_transcript_segment"
            else:
                duration = None
                source = "unavailable"

        return {
            "seconds": round(duration, 2) if duration is not None else None,
            "source": source,
            "is_full_meeting_duration": source == "metadata",
        }

    
    def _extract_cv_summary(self, cv_results: dict) -> dict:
        """
        Read the actual nested results returned by CVPipeline.
        Keep observations from each CV module separate.
        """

        person_results = cv_results.get("person_detection", [])
        scene_results = cv_results.get("scene_analysis", {})
        activity_results = cv_results.get("activity_recognition", {})

        scene_summary = scene_results.get("summary", {})
        activity_summary = activity_results.get("summary", {})

        # Count sampled frames where the person detector found people.
        person_frames_with_people = sum(
            1
            for frame in person_results
            if isinstance(frame, dict)
            and frame.get("people_count", 0) > 0
        )

        return {
            "total_frames": scene_summary.get("frames_analyzed"),
            "frames_with_people": person_frames_with_people,
            "frames_with_faces": activity_summary.get(
                "frames_with_visible_faces"
            ),
            "possible_group_gathering": activity_summary.get(
                "frames_with_possible_group_gathering"
            ),
            "ppe_predictions": scene_summary.get(
                "ppe_detections", {}
            ),
            "attendance_estimate": {
                "value": None,
                "meaning": (
                    "Unique attendance is unknown. People detected "
                    "across sampled frames may be the same individuals."
                ),
            },
            "attention": {
                "value": None,
                "meaning": (
                    "Not determined. Face visibility and head pose "
                    "cannot establish attention."
                ),
            },
            "supervisor_identified": {
                "value": None,
                "meaning": (
                    "Not determined. The current pipeline does not "
                    "reliably identify a person's supervisor role."
                ),
            },
            "module_summaries": {
                "scene_analysis": scene_summary,
                "activity_recognition": activity_summary,
            },
            "limitations": cv_results.get("limitations", []),
        }


    def _build_limitations(
        self,
        transcript_text: str,
        cv_results: dict,
        metadata: dict,
    ) -> list:
        """
        Record important limitations alongside extracted results.
        """

        limitations = [
            "Keyword matches indicate mentions, not proof that a topic was explained correctly.",
            "Question detection is approximate and does not identify the speaker.",
            "PPE model outputs are predictions and may contain false positives or false negatives.",
            "People detected across multiple frames must not be added together as unique attendees.",
            "Face visibility and head pose do not prove attention.",
            "A transcript may contain speech-recognition errors or omissions.",
        ]

        if not transcript_text.strip():
            limitations.append(
                "No usable transcript was provided; transcript-based topics and questions may be incomplete."
            )

        if not cv_results:
            limitations.append(
                "No computer vision summary was provided."
            )

        if "duration_seconds" not in metadata:
            limitations.append(
                "Full meeting duration requires video/audio metadata; transcript timestamps may cover only part of the recording."
            )

        return limitations


    


