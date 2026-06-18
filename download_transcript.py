"""Download a stand-up transcript from scrapsfromtheloft and clean it,
keeping inline [laughter]-style markers so we know where the audience laughed.

Output: data/<slug>.txt  (cleaned transcript, markers preserved inline)
"""
import re, sys, html, os, urllib.request

URLS = {
    "bill-burr-drop-dead-years":
        "https://scrapsfromtheloft.com/movies/bill-burr-drop-dead-years-transcript/",
    "john-mulaney-kid-gorgeous":
        "https://scrapsfromtheloft.com/comedy/john-mulaney-kid-gorgeous-at-radio-city-full-transcript/",
    "michelle-wolf-the-well":
        "https://scrapsfromtheloft.com/comedy/michelle-wolf-the-well-transcript/",
}

# Any bracketed annotation that mentions laughing/laughter/laughs/chuckle.
LAUGH_RE = re.compile(r"\[[^\]]*\b(?:laughter|laughing|laughs|laugh|chuckl)\w*[^\]]*\]", re.IGNORECASE)
# Any other bracketed stage direction ([normal], [applause], [angry voice], ...).
OTHER_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def extract_paragraphs(page):
    # Narrow to the article body if we can find it (WordPress entry-content).
    m = re.search(r'<div[^>]*class="[^"]*entry-content[^"]*"[^>]*>(.*?)</div>\s*<(?:footer|/article)',
                  page, re.S | re.I)
    body = m.group(1) if m else page
    paras = re.findall(r"<p[^>]*>(.*?)</p>", body, re.S | re.I)
    out = []
    for p in paras:
        p = re.sub(r"<br\s*/?>", " ", p, flags=re.I)
        p = re.sub(r"<[^>]+>", "", p)          # strip remaining tags
        p = html.unescape(p)
        p = p.replace("’", "'").replace("‘", "'")
        p = p.replace("“", '"').replace("”", '"')
        p = p.replace("—", "-").replace("–", "-").replace("…", "...")
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else "bill-burr-drop-dead-years"
    url = URLS[slug]
    print(f"Fetching {url}")
    page = fetch(url)
    paras = extract_paragraphs(page)

    # Drop boilerplate paragraphs (site chrome) before the transcript proper.
    # Keep paragraphs from the first one that contains a laughter/applause marker
    # or is clearly transcript-like, through the last laughter marker.
    laugh_idx = [i for i, p in enumerate(paras) if LAUGH_RE.search(p)]
    if laugh_idx:
        start = max(0, laugh_idx[0] - 2)
        end = min(len(paras), laugh_idx[-1] + 2)
        paras = paras[start:end]

    text = "\n".join(paras)
    # Normalise all laughter variants to a single canonical token [LAUGHTER]
    text = LAUGH_RE.sub(" LAUGHPLACEHOLDER ", text)
    # Remove all remaining bracketed stage directions ([normal], [angry], [applause]...)
    text = OTHER_BRACKET_RE.sub(" ", text)
    text = text.replace("LAUGHPLACEHOLDER", "[LAUGHTER]")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()
    n_laughs = text.count("[LAUGHTER]")

    os.makedirs("data", exist_ok=True)
    path = f"data/{slug}.txt"
    with open(path, "w") as f:
        f.write(text)
    words = len(text.split())
    print(f"Saved {path}: {words} words, {n_laughs} laughter markers, {len(paras)} paragraphs")


if __name__ == "__main__":
    main()
