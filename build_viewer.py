"""Merge all per-model word-level JSONs into a single interactive viewer.html.

Panel: pick a model + an emotion.
  * the joke text is colored by that emotion's activation (z) per word
  * a time-series chart plots the emotion across the whole routine, with the
    audience laughs marked as vertical ticks
  * hovering a word draws a linked vertical cursor on the chart (and vice versa)
"""
import json, glob, os

WORDS = json.load(open("data/joke/_words.json"))
models = {}
for p in sorted(glob.glob("data/joke/*.json")):
    if os.path.basename(p) == "_words.json":
        continue
    d = json.load(open(p))
    models[d["tag"]] = {"probe_layer": d["probe_layer"],
                        "emotions": d["emotions"], "word_scores": d["word_scores"]}

emotions = sorted({e for m in models.values() for e in m["emotions"]})
PRIORITY = ["amused", "playful", "delighted", "mirthful", "surprised", "curious", "bored"]
emotions = [e for e in PRIORITY if e in emotions] + [e for e in emotions if e not in PRIORITY]

DATA = {"words": WORDS["words"], "laugh_after_word": WORDS["laugh_after_word"],
        "transcript": WORDS.get("transcript", ""), "emotions": emotions, "models": models}

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Emotion concepts across a stand-up routine</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --ink:#e8e8ea; --mut:#9aa0aa; --accent:#e2574c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:12px 20px; border-bottom:1px solid #262a33; }
  h1 { font-size:17px; margin:0 0 2px; }
  .sub { color:var(--mut); font-size:12.5px; }
  .wrap { display:flex; height:calc(100vh - 54px); }
  .panel { width:270px; flex:none; background:var(--panel); border-right:1px solid #262a33;
           padding:16px; overflow:auto; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut);
              margin:18px 0 8px; }
  .panel h2:first-child { margin-top:0; }
  label.m { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:7px;
            cursor:pointer; font-size:13.5px; }
  label.m:hover { background:#20242d; }
  label.m input { accent-color:var(--accent); }
  .pl { color:var(--mut); font-size:11px; margin-left:auto; }
  select, .scale { width:100%; background:#11141a; color:var(--ink); border:1px solid #2c313b;
            border-radius:7px; padding:8px; font-size:13.5px; }
  .stat { background:#11141a; border:1px solid #2c313b; border-radius:8px; padding:10px;
          font-size:12.5px; margin-top:8px; }
  .stat b { color:#fff; }
  .legend { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--mut); margin-top:6px; }
  .legend .bar { height:10px; flex:1; border-radius:3px;
     background:linear-gradient(90deg,#2f6fe0,#11141a 50%,#e2574c); }
  table.hm { border-collapse:collapse; margin-top:4px; }
  table.hm td.rl { font-size:10px; color:#cfd3da; text-align:right; padding-right:5px; white-space:nowrap; }
  table.hm td.hc { width:14px; height:13px; border:1px solid #0c0e12; cursor:help; }
  .hmscale { display:flex; align-items:center; gap:6px; font-size:10px; color:var(--mut); margin-top:5px; }
  .hmscale .bar { height:9px; flex:1; border-radius:3px;
     background:linear-gradient(90deg,#2166ac,#11141a 50%,#b2182b); }
  .main { flex:1; display:flex; flex-direction:column; min-width:0; }
  .chartbox { flex:none; border-bottom:1px solid #262a33; background:#0c0e12; padding:8px 12px 4px; }
  #chart { width:100%; height:150px; display:block; cursor:crosshair; }
  .chartlbl { font-size:11px; color:var(--mut); display:flex; justify-content:space-between; }
  .reader { flex:1; overflow:auto; padding:22px 30px; }
  .beat { margin:0 0 2px; }
  .w { padding:1px 0; border-radius:3px; }
  .w.hot { outline:2px solid #fff3; }
  .laugh { display:inline-block; margin:0 4px; padding:0 7px; border-radius:10px;
           background:#3a1d1d; color:#ff8a7a; font-size:11px; font-weight:600; vertical-align:middle; }
  .hint { color:var(--mut); font-size:12px; margin-top:4px; }
</style></head>
<body>
<header>
  <h1>Emotion concepts flowing through a stand-up routine</h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="wrap">
  <div class="panel">
    <h2>Model</h2><div id="models"></div>
    <h2>Emotion</h2><select id="emotion"></select>
    <h2>Intensity scale (z)</h2>
    <input class="scale" id="scale" type="range" min="1" max="4" step="0.5" value="2.5">
    <div class="legend"><span>&minus;z</span><div class="bar"></div><span>+z</span></div>
    <label class="m" style="margin-top:10px"><input type="checkbox" id="showlaugh" checked> show laughter</label>
    <h2>Laugh alignment</h2><div class="stat" id="stat"></div>
    <div class="hint">Color &amp; chart = how strongly the chosen model represents the chosen emotion at each word. Red ticks on the chart = audience laughs. Hover the text to move the chart cursor.</div>
    <h2>Probe &times; scenario (cosine)</h2>
    <div id="heatmap"></div>
    <div class="hmscale"><span>&minus;</span><div class="bar"></div><span>+</span></div>
    <div class="hint">Rows = emotion probe. 12 implicit scenarios (the first 3 are jokes). Hover a cell for names &amp; value. Updates with the selected model.</div>
  </div>
  <div class="main">
    <div class="chartbox">
      <div class="chartlbl"><span id="chartTitle"></span><span id="chartHover"></span></div>
      <canvas id="chart"></canvas>
    </div>
    <div class="reader" id="reader"></div>
  </div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const tags = Object.keys(D.models);
let cur = {model: tags[0], emotion: D.emotions[0], scale: 2.5, showlaugh: true};
const laughSet = new Set(D.laugh_after_word);
const N = D.words.length;

// ---- controls ----
const mdiv = document.getElementById('models');
tags.forEach((t,i)=>{
  const l=document.createElement('label'); l.className='m';
  l.innerHTML=`<input type="radio" name="model" value="${t}" ${i===0?'checked':''}>
     <span>${t}</span><span class="pl">L${D.models[t].probe_layer}</span>`;
  l.querySelector('input').onchange=()=>{cur.model=t; refresh();};
  mdiv.appendChild(l);
});
const esel=document.getElementById('emotion');
D.emotions.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;esel.appendChild(o);});
esel.onchange=()=>{cur.emotion=esel.value; refresh();};
document.getElementById('scale').oninput=e=>{cur.scale=+e.target.value; refresh();};
document.getElementById('showlaugh').onchange=e=>{cur.showlaugh=e.target.checked;
  document.querySelectorAll('.laugh').forEach(x=>x.style.display=cur.showlaugh?'inline-block':'none');
  drawBase();};

// ---- render words once ----
const reader=document.getElementById('reader');
const spans=[];
let beat=document.createElement('div'); beat.className='beat'; reader.appendChild(beat);
D.words.forEach((w,i)=>{
  const s=document.createElement('span'); s.className='w'; s.textContent=w+' '; s.dataset.i=i;
  spans.push(s); beat.appendChild(s);
  if(laughSet.has(i)){
    const c=document.createElement('span'); c.className='laugh'; c.textContent='😂 laughter';
    beat.appendChild(c);
    beat=document.createElement('div'); beat.className='beat'; reader.appendChild(beat);
  }
});

// ---- color heatmap ----
function colorFor(z,scale){
  let t=Math.max(-1,Math.min(1,z/scale));
  if(t>=0) return `rgba(226,87,76,${(t*0.85).toFixed(3)})`;
  return `rgba(47,111,224,${(-t*0.85).toFixed(3)})`;
}
function recolor(){
  const sc=D.models[cur.model].word_scores[cur.emotion];
  for(let i=0;i<spans.length;i++){
    const z=sc?sc[i]:0; spans[i].style.background=colorFor(z,cur.scale);
    spans[i].title=cur.emotion+' z='+(sc?z.toFixed(2):'n/a');
  }
}

// ---- chart ----
const cv=document.getElementById('chart'); const ctx=cv.getContext('2d');
let baseImg=null, CW=0, CH=0, DPR=window.devicePixelRatio||1;
function sizeCanvas(){
  CW=cv.clientWidth; CH=cv.clientHeight;
  cv.width=CW*DPR; cv.height=CH*DPR; ctx.setTransform(DPR,0,0,DPR,0,0);
}
function drawBase(){
  const sc=D.models[cur.model].word_scores[cur.emotion]; if(!sc) return;
  sizeCanvas();
  ctx.clearRect(0,0,CW,CH);
  const mid=CH/2, scale=cur.scale;
  // zero line
  ctx.strokeStyle='#2a2f3a'; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(0,mid); ctx.lineTo(CW,mid); ctx.stroke();
  // laughter ticks
  if(cur.showlaugh){ ctx.strokeStyle='rgba(226,87,76,0.28)'; ctx.lineWidth=1;
    D.laugh_after_word.forEach(j=>{const x=j/N*CW; ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,CH);ctx.stroke();});
  }
  // per-pixel mean line
  ctx.strokeStyle='#7fb0ff'; ctx.lineWidth=1.4; ctx.beginPath();
  for(let px=0;px<CW;px++){
    const a=Math.floor(px/CW*N), b=Math.max(a+1,Math.floor((px+1)/CW*N));
    let s=0,c=0; for(let k=a;k<b&&k<N;k++){s+=sc[k];c++;}
    const z=c?s/c:0; const y=mid-Math.max(-1,Math.min(1,z/scale))*mid*0.92;
    if(px===0)ctx.moveTo(px,y); else ctx.lineTo(px,y);
  }
  ctx.stroke();
  baseImg=ctx.getImageData(0,0,cv.width,cv.height);
}
function drawCursor(wordIdx){
  if(!baseImg) return;
  ctx.putImageData(baseImg,0,0);
  if(wordIdx==null) return;
  const x=wordIdx/N*CW;
  ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(x,0); ctx.lineTo(x,CH); ctx.stroke();
  const sc=D.models[cur.model].word_scores[cur.emotion];
  document.getElementById('chartHover').textContent=
    `“${D.words[wordIdx]}”  z=${sc?sc[wordIdx].toFixed(2):'?'}  (word ${wordIdx}/${N})`;
}

// ---- linked hover: text -> chart ----
reader.addEventListener('mousemove',e=>{
  const t=e.target.closest('.w'); if(!t)return;
  const i=+t.dataset.i; drawCursor(i);
});
reader.addEventListener('mouseleave',()=>{drawCursor(null);document.getElementById('chartHover').textContent='';});
// ---- chart -> text (bonus) ----
cv.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect(); const i=Math.round((e.clientX-r.left)/CW*N);
  const idx=Math.max(0,Math.min(N-1,i)); drawCursor(idx);
  spans.forEach(s=>s.classList.remove('hot'));
  const s=spans[idx]; if(s){s.classList.add('hot');
    if(e.shiftKey) s.scrollIntoView({block:'center'});}
});

function hmColor(v,m){ let t=Math.max(-1,Math.min(1,v/m));
  if(t>=0) return `rgba(178,24,43,${(0.12+0.88*t).toFixed(3)})`;
  return `rgba(33,102,172,${(0.12+0.88*-t).toFixed(3)})`; }
function renderHeatmap(){
  const box=document.getElementById('heatmap');
  const sm=D.models[cur.model].scenario_matrix;
  if(!sm||!sm.values){ box.innerHTML='<span class="hint">not computed for this model yet</span>'; return; }
  let mx=0; sm.values.forEach(r=>r.forEach(v=>{if(Math.abs(v)>mx)mx=Math.abs(v);})); mx=mx||0.1;
  let h='<table class="hm">';
  sm.rows.forEach((r,ri)=>{
    h+='<tr><td class="rl">'+r+'</td>';
    sm.values[ri].forEach((v,cj)=>{
      h+=`<td class="hc" style="background:${hmColor(v,mx)}" title="${r} × ${sm.cols[cj]} = ${v.toFixed(3)}"></td>`;
    });
    h+='</tr>';
  });
  box.innerHTML=h+'</table>';
}
function refresh(){
  recolor(); drawBase(); drawCursor(null); renderHeatmap();
  const sc=D.models[cur.model].word_scores[cur.emotion];
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
  let pre=[]; D.laugh_after_word.forEach(j=>{for(let k=Math.max(0,j-5);k<=j;k++)pre.push(sc[k]);});
  const mPre=mean(pre), mAll=mean(sc), lift=mPre-mAll;
  document.getElementById('stat').innerHTML=
    `<b>${cur.emotion}</b> on <b>${cur.model}</b><br>`+
    `mean z &le;6 words before laughs: <b>${mPre.toFixed(3)}</b><br>`+
    `mean z everywhere: ${mAll.toFixed(3)}<br>`+
    `lift into laughs: <b style="color:${lift>0?'#7ee081':'#ff8a7a'}">${lift>=0?'+':''}${lift.toFixed(3)}</b>`;
  document.getElementById('chartTitle').textContent=`${cur.emotion} · ${cur.model} (L${D.models[cur.model].probe_layer})`;
  document.getElementById('subtitle').textContent=
    `${D.transcript} · ${N} words · ${D.laugh_after_word.length} laughs · ${tags.length} models · ${D.emotions.length} emotions`;
}
window.addEventListener('resize',()=>{drawBase();drawCursor(null);});
refresh();
</script>
</body></html>"""

out = HTML.replace("__DATA__", json.dumps(DATA))
open("viewer.html", "w").write(out)
print(f"wrote viewer.html ({len(out)/1e6:.1f} MB) — {len(models)} models, "
      f"{len(DATA['words'])} words, {len(DATA['emotions'])} emotions")
