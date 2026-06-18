"""Build epiplexity.html — a standalone interactive results page for the E1
(in-context epiplexity) experiment, themed to match index.html.

Reads data/<slug>_epiplexity_e1.npz (per-token s_full / s_short / E1 + laugh
positions), re-derives the per-token strings from the transcript (tokenizer only,
no model), computes the laughter-alignment stats with align_and_report, and emits
a self-contained HTML page with an interactive per-token curve, peri-laughter
averages, the stats table, and the interpretation.

Run:  .venv/bin/python build_epiplexity.py
"""
from __future__ import annotations
import json
import numpy as np
from transformers import AutoTokenizer
import align_and_report as ar

SLUG = "bill-burr-drop-dead-years"
NAME = "Bill Burr — Drop Dead Years"
MODEL = "Qwen/Qwen2.5-1.5B"
TRANSCRIPT = f"data/{SLUG}.txt"
NPZ = f"data/{SLUG}_epiplexity_e1.npz"
PRE = 12

d = np.load(NPZ)
s_full, s_short, e1 = d["s_full"], d["s_short"], d["e1"]
laugh_pos = d["laugh_positions"].tolist()
short_ctx = int(d["short_ctx"])

# re-derive token strings (tokenizer only) to match the saved token stream
tok = AutoTokenizer.from_pretrained(MODEL)
text = open(TRANSCRIPT).read()
ids, token_strs, laugh_pos2 = ar.build_token_stream(text, tok)
assert len(token_strs) == len(e1), f"token mismatch {len(token_strs)} vs {len(e1)}"


def stats(scores):
    perm = ar.permutation_test(scores, laugh_pos, pre_window=PRE)
    det = ar.detection_auc(scores, laugh_pos, pre_window=PRE)
    off, mean_c, _sem, nU = ar.peri_laughter_average(scores, laugh_pos, half=40)
    return {"run_z": perm["observed_pre_laugh_z"], "sigma": perm["z_above_null"],
            "p": perm["p_value"], "auc": det, "off0_z": float(mean_c[len(mean_c) // 2])}


SIG = {"e1": stats(e1), "s_full": stats(s_full), "s_short": stats(s_short)}

# peri-laughter average curves (z-scored), shipped precomputed
def peri(scores):
    off, mean_c, sem_c, nU = ar.peri_laughter_average(scores, laugh_pos, half=40)
    return {"off": off.tolist(), "mean": [round(x, 4) for x in mean_c.tolist()],
            "sem": [round(x, 4) for x in sem_c.tolist()], "n": int(nU)}


PERI = {"e1": peri(e1), "s_full": peri(s_full), "s_short": peri(s_short)}

DATA = {
    "name": NAME, "model": MODEL, "short_ctx": short_ctx,
    "n_tokens": len(e1), "n_laughs": len(laugh_pos),
    "tokens": token_strs,
    "laugh_positions": laugh_pos,
    "e1": [round(float(x), 3) for x in e1],
    "s_full": [round(float(x), 3) for x in s_full],
    "s_short": [round(float(x), 3) for x in s_short],
    "stats": SIG, "peri": PERI, "pre_window": PRE,
}

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epiplexity vs laughter — Judging Humor</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --ink:#e8e8ea; --mut:#9aa0aa; --accent:#e2574c;
          --e1:#a07cff; --full:#7fb0ff; --short:#ffb05c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:10px 20px; border-bottom:1px solid #262a33; display:flex; align-items:center; gap:16px; }
  .brand { font-size:13px; color:var(--mut); letter-spacing:.04em; }
  .brand b { color:var(--ink); font-weight:600; }
  header a { color:#7fb0ff; text-decoration:none; font-size:13px; }
  header a:hover { text-decoration:underline; }
  .inner { max-width:1080px; margin:0 auto; padding:26px 26px 90px; }
  h1 { font-size:26px; margin:0 0 10px; line-height:1.22; }
  h2 { font-size:18px; margin:30px 0 8px; padding-top:10px; border-top:1px solid #20242d; }
  p { margin:10px 0; } .lede { font-size:16.5px; color:#dfe3ea; }
  code { background:#11141a; border:1px solid #2c313b; border-radius:4px; padding:.5px 5px; font-size:13px; }
  .mut { color:var(--mut); } a { color:#7fb0ff; }
  .eq { background:#11141a; border:1px solid #2c313b; border-radius:8px; padding:12px 14px;
        font-size:14px; color:#dfe3ea; margin:12px 0; overflow-x:auto; }
  table { border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }
  th,td { text-align:left; padding:8px 10px; border-bottom:1px solid #20242d; }
  th { color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  td b { color:#fff; }
  .sig { color:#7ee081; } .ns { color:#ff8a7a; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:middle; margin-right:5px; }
  .chartcard { background:#0c0e12; border:1px solid #262a33; border-radius:10px; padding:12px 14px; margin:14px 0; }
  .ctrls { display:flex; flex-wrap:wrap; gap:14px; align-items:center; font-size:13px; margin-bottom:8px; }
  .ctrls label { display:flex; align-items:center; gap:6px; cursor:pointer; }
  .ctrls input[type=checkbox]{ accent-color:var(--accent); }
  .chartlbl { font-size:11.5px; color:var(--mut); display:flex; justify-content:space-between; margin:2px 0; }
  #chart { width:100%; height:230px; display:block; cursor:crosshair; }
  #peri { width:100%; height:230px; display:block; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:820px){ .grid2{ grid-template-columns:1fr; } }
  .key { font-size:12px; color:var(--mut); }
  blockquote { margin:14px 0; padding:8px 16px; border-left:3px solid var(--accent);
               background:#14171e; color:#cfd3da; border-radius:0 7px 7px 0; }
</style></head>
<body>
<header>
  <div class="brand"><b>Judging Humor</b> &nbsp;·&nbsp; epiplexity experiment</div>
  <a href="index.html">&larr; back to viewer</a>
</header>
<div class="inner">
<h1>Can epiplexity predict laughter?</h1>
<p class="lede">A first test of the <a href="https://arxiv.org/abs/2601.03220" target="_blank">epiplexity</a>
hypothesis (Finzi et al. 2026): laughter should track the <em>structured, learnable</em> component
of surprise — not raw surprise. We read <span id="hd"></span> token-by-token with a fixed base model
and ask whether the signal rises into the audience's laughs (laughter markers stripped before the
model &mdash; no leakage).</p>

<div class="eq">
E1(t) = &minus;log p(x<sub>t</sub> | last K tokens) &nbsp;&minus;&nbsp; (&minus;log p(x<sub>t</sub> | all prior tokens))
&nbsp;=&nbsp; <b style="color:var(--short)">s_short</b> &minus; <b style="color:var(--full)">s_full</b>
&nbsp;=&nbsp; <b style="color:var(--e1)">in-context epiplexity</b>
<div class="key" style="margin-top:6px">the nats the long-range setup saves on this token = structured information a bounded reader absorbed
(K = <span id="kk"></span>). Raw surprisal <code>s_full</code> is the H1 baseline.</div>
</div>

<h2>Does the signal rise into the laugh?</h2>
<p class="mut" style="font-size:13px">Mean z over the <span id="pw"></span>-token run-up before each laugh, tested against a 5,000-sample permutation null over random positions; detection AUC treats "within the run-up" as the label.</p>
<table id="stbl"><thead><tr><th>signal</th><th>run-up z</th><th>σ above null</th><th>p-value</th><th>detection AUC</th><th>punchline (offset 0)</th></tr></thead><tbody></tbody></table>

<div class="chartcard">
  <div class="ctrls">
    <label><input type="checkbox" id="c_e1" checked><span class="sw" style="background:var(--e1)"></span>E1 epiplexity</label>
    <label><input type="checkbox" id="c_full" checked><span class="sw" style="background:var(--full)"></span>raw surprisal s_full</label>
    <label><input type="checkbox" id="c_short"><span class="sw" style="background:var(--short)"></span>weak-obs s_short</label>
    <label><input type="checkbox" id="c_laugh" checked>laughter ticks</label>
    <label>smooth <input type="range" id="sm" min="1" max="81" step="2" value="31"></label>
    <label>view all <input type="checkbox" id="c_all" checked></label>
  </div>
  <div class="chartlbl"><span>per-token signal (z-scored, smoothed) across the routine</span><span id="hover"></span></div>
  <canvas id="chart"></canvas>
  <div class="key" id="zoomkey">drag-free: hover to inspect a token · uncheck "view all" then drag the slider below to scan a 700-token window</div>
  <input type="range" id="pan" min="0" max="100" value="0" style="width:100%;display:none;margin-top:6px;accent-color:var(--accent)">
</div>

<div class="grid2">
  <div class="chartcard">
    <div class="chartlbl"><span>peri-laughter average (±40 tokens, z)</span><span class="key">offset 0 = last word before the laugh</span></div>
    <canvas id="peri"></canvas>
  </div>
  <div>
    <h2 style="margin-top:0;border:0">What this run shows</h2>
    <p><b>Raw surprisal rises into laughs; forward epiplexity does not</b> — and that null is the
    theoretically <em>right</em> one. <code>E1</code> asks whether the long setup makes the punchline
    <em>more</em> predictable. For a real joke it doesn't: the punchline is an <b>incongruity</b>,
    surprising <em>given</em> the setup. So E1≈0 at the laugh is the signature of incongruity
    <em>without forward resolution</em>.</p>
    <blockquote>The "getting it" — the resolution that makes incongruity funny — is <b>retrodictive</b>:
    how much seeing the punchline lets you recompress the <em>setup</em>. A forward estimator can't see
    it. That motivates <b>E1-backward</b> as the next run.</blockquote>
    <p class="mut" style="font-size:13px">Sanity check that E1 measures real structure: its highest-scoring
    tokens are callbacks/repetitions (e.g. "Why do you think the Klan…" recurs 3×) — long context makes
    the repeat highly compressible. Structure is found; it just isn't concentrated pre-laugh.</p>
  </div>
</div>

<p class="mut" style="font-size:12.5px;margin-top:26px;padding-top:14px;border-top:1px solid #20242d">
Model: <span id="md"></span> (base, fixed) · transcript: <span id="tx"></span> · offset-0 spike is the
punctuation-dominated punchline boundary (a known confound) · single comedian, no orthogonalization yet —
exploratory. Built by <code>build_epiplexity.py</code> from <code>epiplexity_e1.py</code> output.</p>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const N=D.n_tokens, laughs=D.laugh_positions, laughSet=new Set(laughs);
document.getElementById('hd').textContent=`${D.name} (${N.toLocaleString()} tokens, ${D.n_laughs} laughs)`;
document.getElementById('kk').textContent=D.short_ctx;
document.getElementById('pw').textContent=D.pre_window;
document.getElementById('md').textContent=D.model;
document.getElementById('tx').textContent=D.name;

// z-score helper
function z(a){let m=0;for(const x of a)m+=x;m/=a.length;let s=0;for(const x of a)s+=(x-m)*(x-m);s=Math.sqrt(s/a.length)||1e-8;return a.map(x=>(x-m)/s);}
const Z={e1:z(D.e1), s_full:z(D.s_full), s_short:z(D.s_short)};
const COL={e1:'#a07cff', s_full:'#7fb0ff', s_short:'#ffb05c'};

// ---- stats table ----
const order=[['e1','E1 epiplexity','var(--e1)'],['s_full','raw surprisal s_full','var(--full)'],['s_short','weak-obs s_short','var(--short)']];
const tb=document.querySelector('#stbl tbody');
order.forEach(([k,lbl,c])=>{
  const s=D.stats[k]; const sigcls=s.p<0.05?'sig':'ns';
  const tr=document.createElement('tr');
  tr.innerHTML=`<td><span class="sw" style="background:${c}"></span>${lbl}</td>`+
    `<td>${s.run_z>=0?'+':''}${s.run_z.toFixed(3)}</td>`+
    `<td>${s.sigma>=0?'+':''}${s.sigma.toFixed(1)}σ</td>`+
    `<td class="${sigcls}"><b>${s.p.toFixed(4)}</b>${s.p<0.05?'':' (n.s.)'}</td>`+
    `<td>${s.auc.toFixed(3)}</td>`+
    `<td>${s.off0_z>=0?'+':''}${s.off0_z.toFixed(2)}</td>`;
  tb.appendChild(tr);
});

// ---- smoothing ----
function smooth(a,k){if(k<=1)return a.slice();const h=(k-1)/2,out=new Array(a.length);let sum=0;const q=[];
  for(let i=0;i<a.length+h;i++){if(i<a.length){sum+=a[i];q.push(a[i]);}if(q.length>k){sum-=q.shift();}
    const idx=i-h;if(idx>=0)out[idx]=sum/q.length;}return out;}

// ---- main chart ----
const cv=document.getElementById('chart'),cx=cv.getContext('2d');
let CW=0,CH=0,DPR=window.devicePixelRatio||1;
const sel=()=>({e1:c_e1.checked,s_full:c_full.checked,s_short:c_short.checked});
function win(){ if(c_all.checked)return[0,N]; const w=700; const s=Math.round(pan.value/100*(N-w)); return [Math.max(0,s),Math.min(N,s+w)];}
function size(){CW=cv.clientWidth;CH=cv.clientHeight;cv.width=CW*DPR;cv.height=CH*DPR;cx.setTransform(DPR,0,0,DPR,0,0);}
let SM={};
function recompute(){const k=+sm.value;SM={e1:smooth(Z.e1,k),s_full:smooth(Z.s_full,k),s_short:smooth(Z.s_short,k)};}
function draw(){
  size(); if(!CW)return; cx.clearRect(0,0,CW,CH);
  const [a,b]=win(), span=b-a, mid=CH/2, sc=2.6;
  cx.strokeStyle='#2a2f3a';cx.lineWidth=1;cx.beginPath();cx.moveTo(0,mid);cx.lineTo(CW,mid);cx.stroke();
  if(c_laugh.checked){cx.strokeStyle='rgba(226,87,76,0.30)';cx.lineWidth=1;
    laughs.forEach(j=>{if(j<a||j>=b)return;const x=(j-a)/span*CW;cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,CH);cx.stroke();});}
  const s=sel();
  for(const k of ['s_short','s_full','e1']){ if(!s[k])continue;
    const arr=SM[k]; cx.strokeStyle=COL[k]; cx.lineWidth=k==='e1'?1.6:1.1; cx.globalAlpha=k==='e1'?1:0.85; cx.beginPath();
    for(let px=0;px<CW;px++){const i0=a+Math.floor(px/CW*span),i1=Math.max(i0+1,a+Math.floor((px+1)/CW*span));
      let sum=0,c=0;for(let i=i0;i<i1&&i<b;i++){sum+=arr[i];c++;}const v=c?sum/c:0;
      const y=mid-Math.max(-1,Math.min(1,v/sc))*mid*0.94; if(px===0)cx.moveTo(px,y);else cx.lineTo(px,y);}
    cx.stroke(); cx.globalAlpha=1;
  }
}
function hover(clientX){const r=cv.getBoundingClientRect();const [a,b]=win(),span=b-a;
  const i=Math.max(a,Math.min(b-1,a+Math.round((clientX-r.left)/CW*span)));
  const near=laughs.some(p=>p>=i&&p-i<15);
  document.getElementById('hover').innerHTML=
    `tok ${i} “${(D.tokens[i]||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}” · `+
    `<span style="color:${COL.e1}">E1 ${D.e1[i]>=0?'+':''}${D.e1[i]}</span> · `+
    `<span style="color:${COL.s_full}">surp ${D.s_full[i]}</span>`+
    (near?' · <span style="color:#ff8a7a">↳laugh≤15</span>':'');
}
cv.addEventListener('mousemove',e=>hover(e.clientX));
[c_e1,c_full,c_short,c_laugh].forEach(el=>el.onchange=draw);
sm.oninput=()=>{recompute();draw();};
c_all.onchange=()=>{pan.style.display=c_all.checked?'none':'block';draw();};
pan.oninput=draw;

// ---- peri-laughter ----
const pv=document.getElementById('peri'),px=pv.getContext('2d');
function drawPeri(){const W=pv.clientWidth,H=pv.clientHeight;pv.width=W*DPR;pv.height=H*DPR;px.setTransform(DPR,0,0,DPR,0,0);
  px.clearRect(0,0,W,H);const off=D.peri.e1.off,n=off.length;
  let lo=1e9,hi=-1e9;['e1','s_full','s_short'].forEach(k=>D.peri[k].mean.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  const pad=(hi-lo)*0.12||1;lo-=pad;hi+=pad;
  const X=i=>i/(n-1)*(W-44)+34, Y=v=>H-20-(v-lo)/(hi-lo)*(H-34);
  px.strokeStyle='#2a2f3a';px.beginPath();px.moveTo(34,Y(0));px.lineTo(W-10,Y(0));px.stroke();
  const x0=X((n-1)/2);px.strokeStyle='rgba(226,87,76,0.6)';px.lineWidth=1.2;px.beginPath();px.moveTo(x0,4);px.lineTo(x0,H-16);px.stroke();
  px.fillStyle='#9aa0aa';px.font='10px sans-serif';px.fillText('0',x0-3,H-4);px.fillText('-40',30,H-4);px.fillText('+40',W-26,H-4);
  ['s_short','s_full','e1'].forEach(k=>{const m=D.peri[k].mean;px.strokeStyle=COL[k];px.lineWidth=k==='e1'?2:1.3;px.beginPath();
    m.forEach((v,i)=>{const x=X(i),y=Y(v);i?px.lineTo(x,y):px.moveTo(x,y);});px.stroke();});
}

function redraw(){recompute();draw();drawPeri();}
window.addEventListener('resize',redraw);
recompute();draw();drawPeri();
</script>
</body></html>"""

out = HTML.replace("__DATA__", json.dumps(DATA))
open("epiplexity.html", "w").write(out)
print(f"wrote epiplexity.html ({len(out)/1e6:.2f} MB) — {len(token_strs)} tokens, "
      f"{len(laugh_pos)} laughs; E1 p={SIG['e1']['p']:.4f}, s_full p={SIG['s_full']['p']:.4f}")
