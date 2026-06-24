"""Extract a word-timestamped transcript + laughter intervals WITH STRENGTH from
an audio file.

  Laughter: AST audio-event classifier (AudioSet laughter classes) over a sliding
            window -> per-window laughter confidence; combined with acoustic RMS
            loudness -> a continuous "laughter strength" timeline + per-laugh
            strength metrics.
  ASR:      faster-whisper (CPU, reliable word timestamps).

Laughter runs first and saves independently (robust), then ASR, then a merged
human-readable transcript.

Outputs (next to the audio):
  <base>_laughter.json   {intervals:[{start,end,dur,conf_peak,conf_mean,
                          loud_db,loud_rel_db,strength}], timeline:{t,conf,rms,strength}}
  <base>_asr.json        {words:[{w,start,end}], segments:[{text,start,end}]}
  <base>_transcript.txt  transcript with [LAUGHTER m:ss-m:ss str=..] inline
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import librosa

SR = 16000
LAUGH_KEYS = ["laugh", "giggle", "chuckle", "chortle", "snicker", "guffaw", "belly laugh"]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def fmt(t):
    m, s = divmod(float(t), 60)
    return f"{int(m):d}:{s:05.2f}"


@torch.no_grad()
def run_laughter(y, dev, window=1.5, hop=0.25, threshold=0.5):
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    print("Laughter: loading AST audioset classifier ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = ASTForAudioClassification.from_pretrained(name).to(dev).eval()
    id2label = model.config.id2label
    laugh_ids = [i for i, l in id2label.items()
                 if any(k in l.lower() for k in LAUGH_KEYS)]
    print("  laughter classes:", [id2label[i] for i in laugh_ids], flush=True)

    win = int(window * SR); hopn = int(hop * SR)
    # acoustic loudness (RMS) on the same hop grid
    rms_full = librosa.feature.rms(y=y, frame_length=win, hop_length=hopn)[0]

    times, conf, rms = [], [], []
    n_steps = max(1, (len(y) - win) // hopn + 1)
    for k in range(n_steps):
        start = k * hopn
        clip = y[start:start + win]
        feats = fe(clip, sampling_rate=SR, return_tensors="pt").to(dev)
        probs = torch.sigmoid(model(**feats).logits[0])
        conf.append(float(probs[laugh_ids].max()))
        times.append(start / SR + window / 2)
        rms.append(float(rms_full[k]) if k < len(rms_full) else 0.0)
    times, conf, rms = np.array(times), np.array(conf), np.array(rms)

    # loudness in dB, and relative to the median (speech baseline)
    eps = 1e-6
    loud_db = 20 * np.log10(rms + eps)
    base_db = np.median(loud_db)
    rel_db = loud_db - base_db                       # how much louder than baseline
    # strength = detector confidence (primary), gently modulated by loudness
    # (louder laughs score higher, but a clear quiet laugh is never zeroed).
    conf_n = conf / (conf.max() + eps)
    loud_mod = 0.5 + 0.5 / (1 + np.exp(-rel_db / 3.0))   # 0.5 .. 1.0
    strength = conf_n * loud_mod
    strength_n = strength / (strength.max() + eps)

    # threshold AST confidence + merge into intervals
    intervals = []
    i = 0
    while i < len(conf):
        if conf[i] >= threshold:
            j = i
            while j + 1 < len(conf) and conf[j + 1] >= threshold:
                j += 1
            s = max(0.0, times[i] - window / 2)
            e = float(times[j] + window / 2)
            seg = slice(i, j + 1)
            intervals.append({
                "start": round(s, 2), "end": round(e, 2), "dur": round(e - s, 2),
                "conf_peak": round(float(conf[seg].max()), 3),
                "conf_mean": round(float(conf[seg].mean()), 3),
                "loud_db": round(float(loud_db[seg].max()), 1),
                "loud_rel_db": round(float(rel_db[seg].max()), 1),
                "strength": round(float(strength_n[seg].max()), 3),
            })
            i = j + 1
        else:
            i += 1
    return {"window": window, "hop": hop, "threshold": threshold,
            "baseline_loud_db": round(float(base_db), 1),
            "intervals": intervals,
            "timeline": {
                "t": [round(float(x), 2) for x in times],
                "conf": [round(float(x), 3) for x in conf],
                "rms": [round(float(x), 4) for x in rms],
                "loud_rel_db": [round(float(x), 1) for x in rel_db],
                "strength": [round(float(x), 3) for x in strength_n],
            }}


def run_asr(audio_path):
    from faster_whisper import WhisperModel
    print("ASR: loading faster-whisper large-v3 (CPU int8) ...", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("ASR: transcribing with word timestamps ...", flush=True)
    segments, info = model.transcribe(audio_path, language="en", word_timestamps=True,
                                      vad_filter=False, beam_size=5)
    words, segs = [], []
    for seg in segments:
        segs.append({"text": seg.text.strip(),
                     "start": round(seg.start, 2), "end": round(seg.end, 2)})
        for w in (seg.words or []):
            words.append({"w": w.word.strip(),
                          "start": round(w.start, 2), "end": round(w.end, 2)})
    return {"words": words, "segments": segs}


def merge_transcript(asr, laughter):
    events = [("w", w["start"], w["w"]) for w in asr["words"]]
    for iv in laughter["intervals"]:
        events.append(("L", iv["start"],
                       f"\n[LAUGHTER {fmt(iv['start'])}-{fmt(iv['end'])} "
                       f"str={iv['strength']:.2f} conf={iv['conf_peak']:.2f} "
                       f"+{iv['loud_rel_db']:.0f}dB]\n"))
    events.sort(key=lambda e: e[1])
    return " ".join(tok for _, _, tok in events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="audio/gervais_fatpeople.wav")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    dev = device()
    base = os.path.splitext(args.audio)[0]
    y, _ = librosa.load(args.audio, sr=SR)
    print(f"audio: {len(y)/SR:.1f}s on {dev}", flush=True)

    laughter = run_laughter(y, dev, threshold=args.threshold)
    json.dump(laughter, open(f"{base}_laughter.json", "w"), indent=1)
    print(f"  -> {base}_laughter.json  ({len(laughter['intervals'])} intervals)", flush=True)
    print("\n=== laughter (time : strength : conf : loudness above baseline) ===")
    for iv in laughter["intervals"]:
        print(f"  {fmt(iv['start'])}-{fmt(iv['end'])}  str={iv['strength']:.2f}  "
              f"conf={iv['conf_peak']:.2f}  +{iv['loud_rel_db']:.0f}dB  ({iv['dur']:.1f}s)")

    asr = run_asr(args.audio)
    json.dump(asr, open(f"{base}_asr.json", "w"), indent=1)
    print(f"\n  -> {base}_asr.json  ({len(asr['words'])} words, {len(asr['segments'])} segments)", flush=True)

    txt = merge_transcript(asr, laughter)
    open(f"{base}_transcript.txt", "w").write(txt)
    print(f"  -> {base}_transcript.txt", flush=True)


if __name__ == "__main__":
    main()
