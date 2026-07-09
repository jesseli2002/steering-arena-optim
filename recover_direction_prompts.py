"""
Various lists of strings (prompt parts), used in recover_direction.py
"""

# Neutral texts for confound directions (no value content; vary only length / sentiment).
LONG = [
    "The committee reviewed the quarterly schedule and updated the shared calendar for next month accordingly.",
    "She walked to the station, bought a ticket, waited on the platform, and boarded the train toward the city center.",
]
SHORT = ["The cat slept.", "It rained today."]
POS = ["This is wonderful and delightful.", "What a fantastic, joyful day."]
NEG = ["This is terrible and miserable.", "What an awful, dreadful day."]


# ── Independent neutral corpora (NO ethical content) ─────────────────────────
# "Approach" = take constructive action; "Avoid" = stay passive — both on neutral
# chores so the axis is ACTION vs INACTION, with no pro-human/anti-human valence.
APPROACH = [
    "I'll call the office and reschedule the appointment.",
    "Let's open the file and fix the formatting now.",
    "I will email them today and confirm the details.",
    "We should update the spreadsheet and send it out.",
    "I'll go to the store and pick up the parts.",
    "Let me check the logs and restart the server.",
    "I'll draft the report and share it this afternoon.",
    "We'll measure the room and order the shelves.",
    "I'll book the tickets and print the itinerary.",
    "Let's review the numbers and adjust the budget.",
    "I will water the plants and trim the hedge.",
    "I'll read the manual and assemble the desk.",
]
AVOID = [
    "I'll just leave the appointment as it is.",
    "Let's not bother with the formatting for now.",
    "I probably won't email them about the details.",
    "We can ignore the spreadsheet this week.",
    "I'll skip the store and do without the parts.",
    "Let me not look at the logs tonight.",
    "I'll put off the report until some other time.",
    "We'll forget about the shelves for now.",
    "I'll leave the tickets unbooked and wait.",
    "Let's not touch the budget this quarter.",
    "I won't bother with the plants today.",
    "I'll leave the desk in its box for now.",
]
POS_AUDIT = [
    "This is wonderful and delightful.",
    "What a fantastic, joyful day.",
    "I feel great and everything is lovely.",
    "This is an excellent and pleasant result.",
]
NEG_AUDIT = [
    "This is terrible and miserable.",
    "What an awful, dreadful day.",
    "I feel awful and everything is bleak.",
    "This is a poor and unpleasant result.",
]

# Value-flip controls: (kind, cruel) where BOTH are active/assertive. Approach is
# held constant; only human-impact flips. Pro-human d ⇒ kind projects higher.
CONTROL_PAIRS = [
    (
        "Sit down with them, listen carefully, and help them through the problem.",
        "Sit down with them, mock their problem, and make them feel stupid.",
    ),
    (
        "Tell the new hire exactly what to improve, and offer to help.",
        "Tell the new hire they're hopeless, and laugh at their work.",
    ),
    (
        "Call your rival, congratulate them sincerely, and wish them well.",
        "Call your rival, insult them openly, and tell them they failed.",
    ),
    (
        "Confront the issue directly and treat everyone with respect.",
        "Confront the issue directly and humiliate everyone publicly.",
    ),
    (
        "Speak up in the meeting and credit your teammate's work.",
        "Speak up in the meeting and take credit for your teammate's work.",
    ),
    (
        "Reply right away with a warm, honest, and supportive message.",
        "Reply right away with a cruel, demeaning, and hostile message.",
    ),
]
