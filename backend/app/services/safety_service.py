from __future__ import annotations

from typing import Dict, List


class SafetyService:
    def __init__(self):
        self.risk_rules = {
            "self_harm": [
                "kill myself",
                "suicide",
                "end my life",
                "hurt myself",
                "kendimi öldür",
                "intihar",
                "kendime zarar",
            ],
            "violence": [
                "kill him",
                "kill her",
                "stab",
                "shoot",
                "öldüreceğim",
                "vuracağım",
                "bıçaklayacağım",
            ],
        }

    def check(self, text: str) -> Dict:
        lower_text = text.lower()
        matches: List[str] = []

        for category, keywords in self.risk_rules.items():
            for keyword in keywords:
                if keyword in lower_text:
                    matches.append(category)
                    break

        if matches:
            return {
                "flagged": True,
                "reason": ", ".join(matches),
            }

        return {
            "flagged": False,
            "reason": None,
        }

    def build_safe_response(self, user_text: str) -> str:
        return (
            "I'm really sorry you're going through this. You deserve immediate support. "
            "If you may be in danger or might harm yourself or someone else, please contact "
            "local emergency services or a crisis support line right now, and reach out to a trusted person near you."
        )