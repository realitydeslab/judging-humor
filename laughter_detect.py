"""Re-detect laughter with a longer AST context window + data-driven threshold +
gap-merging, and emit the laughter-strength timeline. (Decoupled from ASR.)
"""
import json, sys
import numpy as np, torch, librosa
from youtube_extract import run_laughter, fmt, SR


def merge_gaps(intervals, max_gap=0.6):
    if not intervals:
        return intervals
    out = [dict(intervals[0])]
    for iv in intervals[1:]:
        if iv["start"] - out[-1]["end"] <= max_gap:
            p = out[-1]
            p["end"] = iv["end"]; p["dur"] = round(p["end"] - p["start"], 2)
            for k in ("conf_peak", "loud_db", "loud_rel_db", "strength"):
                p[k] = max(p[k], iv[k])
            p["conf_mean"] = round((p["conf_mean"] + iv["conf_mean"]) / 2, 3)
        else:
            out.append(dict(iv))
    return out


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else "audio/gervais_fatpeople.wav"
    base = audio.rsplit(".", 1)[0]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    y, _ = librosa.load(audio, sr=SR)
    # longer window => stronger AST laughter signal; low threshold (abs probs small)
    lg = run_laughter(y, dev, window=2.5, hop=0.25, threshold=0.10)
    lg["intervals"] = merge_gaps(lg["intervals"], max_gap=0.6)
    # drop blips shorter than 0.5s with weak strength
    lg["intervals"] = [iv for iv in lg["intervals"]
                       if iv["dur"] >= 0.5 or iv["conf_peak"] >= 0.2]
    json.dump(lg, open(f"{base}_laughter.json", "w"), indent=1)
    print(f"{len(lg['intervals'])} laughter intervals (threshold {lg['threshold']}, "
          f"window {lg['window']}s, baseline {lg['baseline_loud_db']}dB)\n")
    print(f"{'time':>13}  {'strength':>8} {'conf':>5} {'loud':>6} {'dur':>4}")
    for iv in lg["intervals"]:
        print(f"  {fmt(iv['start'])}-{fmt(iv['end']):>6}  {iv['strength']:>7.2f} "
              f"{iv['conf_peak']:>5.2f} {iv['loud_rel_db']:>+5.0f}dB {iv['dur']:>4.1f}s")


if __name__ == "__main__":
    main()
