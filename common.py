"""Shared text cleaning + routing config (used by train.py and app.py)."""
import re

TEAM_MAP = {
    "Billing": "Billing & Finance Team",
    "Refund Request": "Billing & Finance Team",
    "Technical Issue": "Technical Support Team",
    "Account Access": "Account Security Team",
    "Product Inquiry": "Sales & Product Team",
    "Cancellation": "Customer Retention Team",
}
DEFAULT_TEAM = "General Support Team"
PRIORITIES = ["critical", "high", "medium", "low"]
CONF_THRESHOLD = 0.55  # below this -> manual queue


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"<[^>]+>", " ", t)          # strip HTML
    t = re.sub(r"http\S+", " ", t)          # strip urls
    t = re.sub(r"[^a-z0-9\s]", " ", t)      # strip punctuation
    return re.sub(r"\s+", " ", t).strip()
