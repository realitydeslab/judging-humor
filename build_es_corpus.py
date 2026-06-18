"""Build EmotionScope's consolidated template corpus, adding a 21st emotion:
AMUSED.  EmotionScope computes each emotion vector as (emotion mean - grand mean
over ALL emotions), so 'amused' only gets a meaningful direction when it sits
alongside the other 20 emotions.  We therefore merge the repo's 20 story files
with our amused passages and write the default templates path the extractor reads
(data/templates/emotion_stories.jsonl).
"""
import json, glob, os
import emotion_data  # our funny corpus (POS) -> "amused"
import humor_data    # playful / delighted / mirthful / bored

ES = "EmotionScope"
STORY_DIR = f"{ES}/data/story_contributions"
OUT = f"{ES}/data/templates/emotion_stories.jsonl"

# Longer narrative amused templates, matched in style to the repo's story corpus
# (their entries are single long sentences), to complement the punchy one-liners.
AMUSED_EXTRA = [
    "She tried to explain the board game rules with total authority, got every single one wrong, and only realized halfway through when her nephew quietly asked why the dragon was allowed to pay rent.",
    "The waiter kept a completely straight face while describing the soup, then leaned in and whispered that it was, in his professional opinion, just very confident water, and walked off before anyone could respond.",
    "He spent forty minutes assembling the bookshelf, stepped back to admire it, and watched in slow motion as it folded itself flat again like it had simply changed its mind about being furniture.",
    "My grandmother insisted she wasn't competitive, then absolutely demolished an eight-year-old at mini golf and did a small, dignified victory shimmy by the windmill hole.",
    "The cat sat on the laptop, sent a forty-page document to my entire contacts list consisting of the single letter 'p', and then looked at me as though I were the one being unreasonable.",
    "The instructor said the yoga pose would bring inner peace, and the entire row of us toppled over like dominoes while she serenely held it and pretended not to notice the carnage.",
    "He confidently told the tour group that the painting was priceless and centuries old, and the actual museum guide gently pointed out it was a fire exit sign.",
    "I asked the toddler what the dog said and she gave a fifteen-minute answer involving three voices, a plot twist, and an intermission, and honestly it was the best story I've heard all year.",
    "The GPS lady clearly gave up on me somewhere around the third roundabout and just started saying 'sure, why not' every time I made a turn.",
    "My dad bought a label maker and within an hour everything in the house had a label, including the label maker, which now proudly reads 'label maker'.",
]


def main():
    rows = []
    for path in sorted(glob.glob(f"{STORY_DIR}/stories_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    rows.append({"emotion": e["emotion"], "text": e["text"]})
    n_base = len(rows)
    base_emotions = sorted(set(r["emotion"] for r in rows))

    added = {}
    amused = list(emotion_data.POS) + AMUSED_EXTRA
    for t in amused:
        rows.append({"emotion": "amused", "text": t})
    added["amused"] = len(amused)

    for name, templates in humor_data.TEMPLATES.items():
        for t in templates:
            rows.append({"emotion": name, "text": t})
        added[name] = len(templates)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{n_base} base rows across {len(base_emotions)} emotions: {base_emotions}")
    print(f"+ new emotions: {added}")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
