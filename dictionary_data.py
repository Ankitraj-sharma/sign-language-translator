"""
Dictionary Data for Sign Language Recognition.
Defines supported signs, visual cues, categories, and instructions for users.
"""

from typing import TypedDict


class SignEntry(TypedDict):
    label: str
    category: str
    emoji: str
    meaning: str
    how_to_perform: str
    finger_status: str


SIGN_DICTIONARY: dict[str, SignEntry] = {
    # ── Common Words & Phrases ──────────────────────────────────────────
    "HELLO": {
        "label": "HELLO",
        "category": "Greetings",
        "emoji": "👋",
        "meaning": "Friendly greeting or salutation",
        "how_to_perform": "Open palm with all 5 fingers fully extended and spread out facing the camera.",
        "finger_status": "All 5 fingers extended up",
    },
    "THANK YOU": {
        "label": "THANK YOU",
        "category": "Greetings",
        "emoji": "🙏",
        "meaning": "Expression of gratitude",
        "how_to_perform": "Flat open hand tilted slightly forward from near the chin towards the screen.",
        "finger_status": "All fingers together, angled towards viewer",
    },
    "YES": {
        "label": "YES",
        "category": "Responses",
        "emoji": "👍",
        "meaning": "Agreement or affirmation",
        "how_to_perform": "Make a fist with your thumb pointing upright (Thumbs Up).",
        "finger_status": "Thumb UP, other 4 fingers curled in fist",
    },
    "NO": {
        "label": "NO",
        "category": "Responses",
        "emoji": "👎",
        "meaning": "Disagreement, refusal, or negation",
        "how_to_perform": "Fist with thumb pointing directly downwards (Thumbs Down).",
        "finger_status": "Thumb DOWN, other 4 fingers curled in fist",
    },
    "I LOVE YOU": {
        "label": "I LOVE YOU",
        "category": "Expressions",
        "emoji": "🤟",
        "meaning": "Universal ASL symbol for 'I Love You'",
        "how_to_perform": "Extend Thumb, Index, and Pinky fingers while keeping Middle and Ring fingers curled.",
        "finger_status": "Thumb, Index, Pinky extended; Middle, Ring curled",
    },
    "PEACE / VICTORY": {
        "label": "PEACE / VICTORY",
        "category": "Expressions",
        "emoji": "✌️",
        "meaning": "Peace sign, victory, or number 2",
        "how_to_perform": "Extend Index and Middle fingers in a V shape; keep Thumb, Ring, Pinky curled.",
        "finger_status": "Index & Middle extended in V; others curled",
    },
    "OKAY": {
        "label": "OKAY",
        "category": "Responses",
        "emoji": "👌",
        "meaning": "Approval, agreement, everything is fine",
        "how_to_perform": "Form a circle with Thumb and Index tip; extend Middle, Ring, Pinky up.",
        "finger_status": "Thumb + Index circle; 3 fingers extended",
    },
    "HELP": {
        "label": "HELP",
        "category": "Emergency & Needs",
        "emoji": "🆘",
        "meaning": "Urgent request for assistance or aid",
        "how_to_perform": "Closed fist held straight or resting on an open flat palm.",
        "finger_status": "Tight fist held at chest level",
    },
    "STOP": {
        "label": "STOP",
        "category": "Emergency & Needs",
        "emoji": "🛑",
        "meaning": "Cease action, wait, or halt",
        "how_to_perform": "Hold an open vertical palm facing directly outward towards the camera.",
        "finger_status": "Vertical palm, fingers straight up",
    },
    "WATER": {
        "label": "WATER",
        "category": "Emergency & Needs",
        "emoji": "💧",
        "meaning": "Requesting drinking water",
        "how_to_perform": "Form a 'W' shape with Index, Middle, and Ring fingers extended upright.",
        "finger_status": "Index, Middle, Ring extended ('W' sign)",
    },
    "SUPER / CALL ME": {
        "label": "SUPER / CALL ME",
        "category": "Expressions",
        "emoji": "🤙",
        "meaning": "Shaka / Call me / Chill sign",
        "how_to_perform": "Extend Thumb and Pinky outward while keeping 3 middle fingers curled in fist.",
        "finger_status": "Thumb and Pinky extended; others curled",
    },
    "POINT / YOU": {
        "label": "POINT / YOU",
        "category": "Pronouns",
        "emoji": "👉",
        "meaning": "Referencing person, object, or direction",
        "how_to_perform": "Extend only the Index finger; keep Thumb and other fingers curled.",
        "finger_status": "Index pointing; others curled",
    },
    # ── Numbers ─────────────────────────────────────────────────────────
    "ONE (1)": {
        "label": "ONE (1)",
        "category": "Numbers",
        "emoji": "1️⃣",
        "meaning": "Number 1",
        "how_to_perform": "Extend only the Index finger pointing straight up.",
        "finger_status": "Index up only",
    },
    "TWO (2)": {
        "label": "TWO (2)",
        "category": "Numbers",
        "emoji": "2️⃣",
        "meaning": "Number 2",
        "how_to_perform": "Extend Index and Middle fingers straight up together.",
        "finger_status": "Index & Middle up",
    },
    "THREE (3)": {
        "label": "THREE (3)",
        "category": "Numbers",
        "emoji": "3️⃣",
        "meaning": "Number 3",
        "how_to_perform": "Extend Thumb, Index, and Middle fingers up.",
        "finger_status": "Thumb, Index, Middle extended",
    },
    "FOUR (4)": {
        "label": "FOUR (4)",
        "category": "Numbers",
        "emoji": "4️⃣",
        "meaning": "Number 4",
        "how_to_perform": "Extend all four fingers (Index, Middle, Ring, Pinky) with thumb tucked.",
        "finger_status": "4 fingers up; Thumb tucked",
    },
    "FIVE (5)": {
        "label": "FIVE (5)",