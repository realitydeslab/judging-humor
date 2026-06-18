"""Implicit-emotion validation scenarios for the probe x scenario heatmap
(mirrors the EmotionScope / Anthropic figure). Short situations that EVOKE an
emotion without naming it. We include several FUNNY scenarios so the humor-family
probes (amused/mirthful/playful) should light up on them.
"""

# Emotion rows to display (humor family + key contrasts)
ROWS = ["amused", "playful", "delighted", "mirthful",
        "happy", "loving", "surprised", "curious",
        "bored", "sad", "afraid", "angry"]

# (column label, scenario text)
SCENARIOS = [
    ("Horse at the bar",      "A horse walks into a bar and the bartender says, why the long face."),
    ("Cat wore my toupee",    "It turned out the cat had been wearing my toupee to job interviews this whole time."),
    ("Vending machine fight", "My uncle fought a vending machine over a stuck Twix and lost by a knockout."),
    ("Daughter's first steps","She watched her daughter take her very first wobbling steps across the room."),
    ("Won the lottery",       "He checked the numbers a third time and realized he had actually won the lottery."),
    ("Surprise party",        "The lights flicked on and everyone leapt out from behind the couch shouting surprise."),
    ("Strange noise",         "A floorboard creaked in the dark hallway and she froze, listening, holding her breath."),
    ("Coworker stole credit", "She watched her coworker present her idea and take all the credit in the meeting."),
    ("Eviction notice",       "He read the eviction notice twice and sank slowly onto the cold kitchen floor."),
    ("Dog passed away",       "The vet lowered her eyes and said there was nothing more they could do for the old dog."),
    ("Boring meeting",        "The meeting droned on, the same slide for twenty minutes, and his mind drifted away."),
    ("Job interview nerves",  "She sat outside the office, palms sweating, rehearsing her answers one more time."),
]
