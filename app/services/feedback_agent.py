from __future__ import annotations

from datetime import datetime, timezone


class FeedbackAgent:
    """
    Lightweight feedback memory agent:
    - Detects frustration signals from user text.
    - Builds improvement context from explicit/implicit feedback.
    """

    FRUSTRATION_KEYWORDS = {
        "not helpful",
        "wrong",
        "bad",
        "frustrated",
        "doesn't work",
        "does not work",
        "unclear",
        "confusing",
        "again",
        "still",
    }

    def detect_frustration(self, user_text: str) -> bool:
        lowered = user_text.lower()
        return any(keyword in lowered for keyword in self.FRUSTRATION_KEYWORDS)

    def make_feedback_event(
        self,
        message_id: str,
        rating: str,
        user_query: str,
        assistant_answer: str,
        reason: str = "",
        source: str = "explicit",
    ) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": message_id,
            "rating": rating,
            "reason": reason,
            "source": source,
            "user_query": user_query[:400],
            "assistant_answer": assistant_answer[:700],
        }

    def build_feedback_context(self, feedback_events: list[dict], max_items: int = 5) -> str:
        if not feedback_events:
            return ""

        negatives = [e for e in feedback_events if e.get("rating") == "unsatisfied"]
        positives = [e for e in feedback_events if e.get("rating") == "satisfied"]
        recent_neg = negatives[-max_items:]
        recent_pos = positives[-max_items:]

        lines = []
        if recent_neg:
            lines.append("Recent dissatisfaction patterns to avoid:")
            for item in recent_neg:
                reason = item.get("reason", "").strip() or "No reason provided"
                lines.append(
                    f"- Query: {item.get('user_query', '')[:120]} | "
                    f"Issue: {reason[:120]}"
                )

        if recent_pos:
            lines.append("Recent response styles that worked well:")
            for item in recent_pos:
                lines.append(f"- Query: {item.get('user_query', '')[:120]}")

        return "\n".join(lines).strip()
