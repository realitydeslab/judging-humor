"""Build index.html: a tabbed single page with three sections.

  1. Intro       — project overview ("Judging Humor": can an LLM predict the joke?)
  2. Viewer      — interactive emotion-concept viewer over a stand-up routine
  3. Literature  — the literature review (rendered from lit-reviews/literature-review.md)

Viewer section:
  * pick a model + an emotion; the joke text is colored by that emotion's
    activation (z) per word (paper-style heatmap)
  * a time-series chart plots the emotion across the routine; audience laughs
    are vertical red ticks; hovering a word draws a linked cursor on the chart
  * a floating "Emotion Probe x Scenario" figure (bottom-left) shows the cosine
    similarity matrix for the selected model, with row/column labels + colorbar
"""
import json, glob, os, re, html as _html

WORDS = json.load(open("data/joke/_words.json"))
models = {}
for p in sorted(glob.glob("data/joke/*.json")):
    if os.path.basename(p).startswith("_"):   # _words.json, _epiplexity.json, …
        continue
    d = json.load(open(p))
    models[d["tag"]] = {"probe_layer": d["probe_layer"],
                        "emotions": d["emotions"],
                        "word_scores": d["word_scores"],
                        "scenario_matrix": d.get("scenario_matrix")}

emotions = sorted({e for m in models.values() for e in m["emotions"]})
PRIORITY = ["amused", "playful", "delighted", "mirthful", "surprised", "curious", "bored"]
emotions = [e for e in PRIORITY if e in emotions] + [e for e in emotions if e not in PRIORITY]

DATA = {"words": WORDS["words"], "laugh_after_word": WORDS["laugh_after_word"],
        "transcript": WORDS.get("transcript", ""), "emotions": emotions, "models": models}


# ---- minimal, dependency-free Markdown -> HTML (enough for our review) ----
def _inline(t):
    t = _html.escape(t, quote=False)
    t = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
    t = re.sub(r'(?<![">])(https?://[^\s<)\]]+)',
               r'<a href="\1" target="_blank" rel="noopener">\1</a>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    return t


def md_to_html(md):
    out, para, mode = [], [], None  # mode in {None,'ul','ol','bq'}

    def flush_para():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    def close_mode():
        nonlocal mode
        flush_para()
        if mode in ("ul", "ol"):
            out.append(f"</{mode}>")
        elif mode == "bq":
            out.append("</blockquote>")
        mode = None

    for raw in md.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            close_mode(); continue
        if re.match(r'^---+\s*$', line):
            close_mode(); out.append("<hr>"); continue
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            close_mode()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>"); continue
        m = re.match(r'^>\s?(.*)$', line)
        if m:
            if mode != "bq":
                close_mode(); out.append("<blockquote>"); mode = "bq"
            para.append(m.group(1)); continue
        m = re.match(r'^\d+\.\s+(.*)$', line)
        if m:
            if mode != "ol":
                close_mode(); out.append("<ol>"); mode = "ol"
            else:
                flush_para()
            out.append("<li>" + _inline(m.group(1)) + "</li>"); continue
        m = re.match(r'^[-*]\s+(.*)$', line)
        if m:
            if mode != "ul":
                close_mode(); out.append("<ul>"); mode = "ul"
            else:
                flush_para()
            out.append("<li>" + _inline(m.group(1)) + "</li>"); continue
        if mode in ("ul", "ol"):
            close_mode()
        para.append(line)
    close_mode()
    return "\n".join(out)


def build_lit_html(path):
    """Render the review and wire in-text (Author, Year) citations to anchor jumps
    into the numbered reference list."""
    md = open(path).read()

    # ---- parse the reference list -> [(n, surname, year)] in document order ----
    refs = []
    if "## References" in md:
        for line in md.split("## References", 1)[1].split("\n"):
            m = re.match(r'^\s*\d+\.\s+(.*)$', line)
            if not m:
                continue
            text = m.group(1)
            sm = re.match(r"([A-Za-z’'`-]+)", text)
            ym = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", text)
            refs.append({"n": len(refs) + 1,
                         "surname": sm.group(1) if sm else None,
                         "year": ym.group(1) if ym else None})

    html = md_to_html(md)

    # ---- split body from the references section ----
    marker = "<h2>References</h2>"
    has_marker = marker in html
    pre, post = html.split(marker, 1) if has_marker else (html, "")

    # give each reference <li> a stable id (ref1..refN), in order
    cnt = [0]
    def _addid(_m):
        cnt[0] += 1
        return f'<li id="ref{cnt[0]}"'
    post = re.sub(r'<li', _addid, post)

    # ---- linkify in-text citations in the body ----
    connector = r"[A-Za-z,.\s’'&;\-]{0,55}?"
    for r in refs:
        s, y, n = r["surname"], r["year"], r["n"]
        if not s or not y:
            continue
        pat = re.compile(r'(?<!\w)(' + re.escape(s) + connector + r'\(?\b' + re.escape(y) + r'\b\)?)')
        pre = pat.sub(lambda m, n=n: f'<a class="cite" href="#ref{n}">{m.group(1)}</a>', pre)

    return pre + (marker + post if has_marker else "")


LIT_HTML = build_lit_html("lit-reviews/literature-review.md")

INTRO_HTML = """
<h1>Judging Humor</h1>
<p class="lede">Can a language model predict the joke? This project treats humor as
<strong>critical unpredictability</strong> — a punchline poised at the <em>edge of chaos</em>
between the too-expected (boring) and the too-random (nonsense) — and uses a model's own
<strong>surprisal / perplexity</strong> as the signal.</p>

<h2>The idea</h2>
<p>Across 2,400 years, humor theory keeps returning to one structure: a <em>disconfirmed
prediction that is safely resolved</em>. The setup builds a confident expectation (low
surprisal, narrowing entropy); the punchline violates it (a sharp surprisal spike) in a way
that can be retroactively made sense of. Information theory lets us measure that violation
directly with <em>&minus;log p</em> over a language model's next-token distribution — the
project's working coinage, <em>&ldquo;epiplexity&rdquo;</em>.</p>

<h2>What we do</h2>
<ul>
  <li><strong>Probe LLM activations</strong> for emotion / humor concept vectors
      (an EmotionScope-style replication), recovering linear directions for amusement,
      surprise, and related concepts.</li>
  <li><strong>Align those signals against real laughter.</strong> We run the probes over
      stand-up transcripts (Bill Burr &mdash; <em>Drop Dead Years</em>; John Mulaney &mdash;
      <em>Kid Gorgeous</em>) whose audience-laughter moments are marked, and ask whether the
      model &ldquo;lights up&rdquo; just before the crowd laughs.</li>
  <li><strong>Compare models</strong> (Gemma, Qwen, &hellip;) and probe layers to see where,
      and how strongly, humor-relevant concepts are represented.</li>
</ul>

<h2>How to use this page</h2>
<ul>
  <li><strong>Viewer</strong> &mdash; pick a model and an emotion. The transcript is colored
      by that concept's activation per word; the chart traces it across the whole routine with
      red ticks marking audience laughs. Hover the text to move the chart cursor, and watch the
      &ldquo;lift into laughs&rdquo; statistic. The floating figure shows the probe&times;scenario
      cosine matrix.</li>
  <li><strong>Literature</strong> &mdash; the full literature review (humor theory &rarr;
      information theory &rarr; computational humor &amp; LLMs &rarr; mechanistic
      interpretability &rarr; criticality), with ~80 verified references and DOIs. The Markdown
      source and raw research strands live in <code>lit-reviews/</code>.</li>
</ul>

<p class="note">Research conducted with AI-assisted multi-agent literature search; citations
verified against primary records. See the Literature tab for the full review and reference list.</p>
"""

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judging Humor — can a model predict the joke?</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --ink:#e8e8ea; --mut:#9aa0aa; --accent:#e2574c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:10px 20px 0; border-bottom:1px solid #262a33; }
  .brand { font-size:13px; color:var(--mut); letter-spacing:.04em; }
  .brand b { color:var(--ink); font-weight:600; }
  nav.tabs { display:flex; gap:4px; margin-top:8px; }
  nav.tabs button { background:none; border:0; color:var(--mut); cursor:pointer;
    font:inherit; font-size:13.5px; padding:9px 14px; border-bottom:2px solid transparent; }
  nav.tabs button:hover { color:var(--ink); }
  nav.tabs button.active { color:#fff; border-bottom-color:var(--accent); }
  .tab { display:none; }
  .tab.active { display:block; }
  /* ---- document tabs (intro + literature) ---- */
  .doc { height:calc(100vh - 86px); overflow:auto; }
  .doc-inner { max-width:820px; margin:0 auto; padding:34px 28px 90px; }
  .doc-inner h1 { font-size:28px; margin:0 0 14px; line-height:1.2; }
  .doc-inner h2 { font-size:19px; margin:30px 0 8px; padding-top:8px; border-top:1px solid #20242d; }
  .doc-inner h3 { font-size:16px; color:#cfd3da; margin:20px 0 6px; font-weight:600; }
  .doc-inner p { margin:10px 0; }
  .doc-inner a { color:#7fb0ff; text-decoration:none; }
  .doc-inner a:hover { text-decoration:underline; }
  .doc-inner a.cite { color:#b59cff; text-decoration:none; cursor:pointer;
                      border-bottom:1px dotted #6b5fa8; }
  .doc-inner a.cite:hover { color:#cdbcff; border-bottom-style:solid; }
  .doc-inner ol li { scroll-margin-top:16px; transition:background .25s; }
  .doc-inner ol li.flash { background:#2a2440; border-radius:6px;
                           box-shadow:0 0 0 6px #2a2440; }
  .doc-inner ul, .doc-inner ol { margin:10px 0; padding-left:24px; }
  .doc-inner li { margin:5px 0; }
  .doc-inner code { background:#11141a; border:1px solid #2c313b; border-radius:4px; padding:.5px 5px; font-size:13px; }
  .doc-inner hr { border:0; border-top:1px solid #20242d; margin:26px 0; }
  .doc-inner blockquote { margin:14px 0; padding:8px 16px; border-left:3px solid var(--accent);
                          background:#14171e; color:#cfd3da; border-radius:0 7px 7px 0; }
  .doc-inner blockquote p { margin:4px 0; }
  .lede { font-size:17px; color:#dfe3ea; }
  .note { color:var(--mut); font-size:13px; margin-top:24px; padding-top:14px; border-top:1px solid #20242d; }
  .doc-inner ol li { margin:6px 0; font-size:13.5px; color:#cdd2da; }
  /* ---- viewer ---- */
  .vsub { color:var(--mut); font-size:12.5px; padding:6px 20px 4px; }
  .wrap { display:flex; height:calc(100vh - 116px); }
  .panel { width:270px; flex:none; background:var(--panel); border-right:1px solid #262a33;
           padding:16px; overflow:auto; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); margin:18px 0 8px; }
  .panel h2:first-child { margin-top:0; }
  label.m { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:7px; cursor:pointer; font-size:13.5px; }
  label.m:hover { background:#20242d; }
  label.m input { accent-color:var(--accent); }
  .pl { color:var(--mut); font-size:11px; margin-left:auto; }
  select, .scale { width:100%; background:#11141a; color:var(--ink); border:1px solid #2c313b;
            border-radius:7px; padding:8px; font-size:13.5px; }
  .stat { background:#11141a; border:1px solid #2c313b; border-radius:8px; padding:10px; font-size:12.5px; margin-top:8px; }
  .stat b { color:#fff; }
  .legend { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--mut); margin-top:6px; }
  .legend .bar { height:10px; flex:1; border-radius:3px; background:linear-gradient(90deg,#2166ac,#f7f7f7 50%,#b2182b); }
  .main { flex:1; display:flex; flex-direction:column; min-width:0; position:relative; }
  .chartbox { flex:none; border-bottom:1px solid #262a33; background:#0c0e12; padding:8px 12px 4px; }
  #chart { width:100%; height:150px; display:block; cursor:crosshair; }
  .chartlbl { font-size:11px; color:var(--mut); display:flex; justify-content:space-between; }
  .reader { flex:1; overflow:auto; padding:22px 30px 40px; }
  .beat { margin:0 0 2px; }
  .w { padding:1px 0; border-radius:3px; }
  .w.hot { outline:2px solid #fff3; }
  .laugh { display:inline-block; margin:0 4px; padding:0 7px; border-radius:10px;
           background:#3a1d1d; color:#ff8a7a; font-size:11px; font-weight:600; vertical-align:middle; }
  .hint { color:var(--mut); font-size:12px; margin-top:4px; }
  /* floating heatmap figure card */
  #hmpanel { position:absolute; left:14px; bottom:14px; background:#fbfbfb; color:#222;
             border-radius:10px; box-shadow:0 8px 30px #000a; padding:8px 10px 6px; z-index:20; user-select:none; }
  #hmpanel .hmhead { display:flex; align-items:center; justify-content:space-between;
             font:600 12.5px -apple-system,sans-serif; color:#9a3328; margin:0 2px 4px; }
  #hmpanel button { background:#eee; border:1px solid #ccc; border-radius:5px; cursor:pointer;
             font-size:11px; padding:1px 7px; color:#333; }
  #hmpanel.collapsed .hmbody { display:none; }
  #hmpanel svg text { font-family:-apple-system,Segoe UI,sans-serif; fill:#333; }
__EPI_STYLE__
</style></head>
<body>
<header>
  <div class="brand"><b>Judging Humor</b> &nbsp;·&nbsp; can a model predict the joke?</div>
  <nav class="tabs">
    <button data-tab="intro" class="active">Intro</button>
    <button data-tab="viewer">Viewer</button>
    <button data-tab="lit">Literature</button>
    __EPI_NAV__
  </nav>
</header>

<section id="tab-intro" class="tab active doc"><div class="doc-inner">__INTRO__</div></section>

<section id="tab-viewer" class="tab">
  <div class="vsub" id="subtitle"></div>
  <div class="wrap">
    <div class="panel">
      <h2>Model</h2><div id="models"></div>
      <h2>Emotion</h2><select id="emotion"></select>
      <h2>Intensity scale (z)</h2>
      <input class="scale" id="scale" type="range" min="1" max="4" step="0.5" value="2.5">
      <div class="legend"><span>&minus;z</span><div class="bar"></div><span>+z</span></div>
      <label class="m" style="margin-top:10px"><input type="checkbox" id="showlaugh" checked> show laughter</label>
      <label class="m"><input type="checkbox" id="showhm" checked> show probe&times;scenario figure</label>
      <h2>Laugh alignment</h2><div class="stat" id="stat"></div>
      <div class="hint">Color &amp; chart = how strongly the chosen model represents the chosen emotion at each word. Red ticks on the chart = audience laughs. Hover the text to move the chart cursor.</div>
    </div>
    <div class="main">
      <div class="chartbox">
        <div class="chartlbl"><span id="chartTitle"></span><span id="chartHover"></span></div>
        <canvas id="chart"></canvas>
      </div>
      <div class="reader" id="reader"></div>
      <div id="hmpanel">
        <div class="hmhead"><span id="hmtitle">Emotion Probe &times; Scenario</span>
          <button id="hmtoggle">hide</button></div>
        <div class="hmbody" id="hmbody"></div>
      </div>
    </div>
  </div>
</section>

<section id="tab-lit" class="tab doc"><div class="doc-inner">__LIT__</div></section>

__EPI_SECTION__

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const tags = Object.keys(D.models);
let cur = {model: tags[0], emotion: D.emotions[0], scale: 2.5};
const laughSet = new Set(D.laugh_after_word);
const N = D.words.length;
const reader=document.getElementById('reader');
const cv=document.getElementById('chart'); const ctx=cv.getContext('2d');
const spans=[];
let baseImg=null, CW=0, CH=0, DPR=window.devicePixelRatio||1;
let viewerReady=false;

// ---- tab switching ----
const tabBtns = document.querySelectorAll('nav.tabs button');
function showTab(name){
  tabBtns.forEach(b=>b.classList.toggle('active', b.dataset.tab===name));
  document.querySelectorAll('.tab').forEach(s=>s.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if(name==='viewer'){ initViewer(); requestAnimationFrame(()=>{drawBase();drawCursor(null);}); }
  if(name==='epi' && window.__epi){ window.__epi.init(); requestAnimationFrame(()=>window.__epi.redraw()); }
}
tabBtns.forEach(b=>b.onclick=()=>showTab(b.dataset.tab));

// ---- in-text citation jumps (smooth scroll + flash the reference) ----
document.getElementById('tab-lit').addEventListener('click',e=>{
  const a=e.target.closest('a.cite'); if(!a) return;
  e.preventDefault();
  const tgt=document.getElementById(a.getAttribute('href').slice(1));
  if(!tgt) return;
  tgt.scrollIntoView({behavior:'smooth', block:'center'});
  tgt.classList.remove('flash'); void tgt.offsetWidth; tgt.classList.add('flash');
  setTimeout(()=>tgt.classList.remove('flash'), 1600);
});

function initViewer(){
  if(viewerReady) return; viewerReady=true;
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
  document.getElementById('showlaugh').onchange=e=>{
    document.querySelectorAll('.laugh').forEach(x=>x.style.display=e.target.checked?'inline-block':'none'); drawBase();};
  document.getElementById('showhm').onchange=e=>{
    document.getElementById('hmpanel').style.display=e.target.checked?'block':'none';};
  document.getElementById('hmtoggle').onclick=()=>{
    const p=document.getElementById('hmpanel'); p.classList.toggle('collapsed');
    document.getElementById('hmtoggle').textContent=p.classList.contains('collapsed')?'show':'hide';};

  // ---- render words once ----
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

  reader.addEventListener('mousemove',e=>{const t=e.target.closest('.w'); if(!t)return; drawCursor(+t.dataset.i);});
  reader.addEventListener('mouseleave',()=>{drawCursor(null);document.getElementById('chartHover').textContent='';});
  cv.addEventListener('mousemove',e=>{
    const r=cv.getBoundingClientRect(); const idx=Math.max(0,Math.min(N-1,Math.round((e.clientX-r.left)/CW*N)));
    drawCursor(idx); spans.forEach(s=>s.classList.remove('hot'));
    const s=spans[idx]; if(s){s.classList.add('hot'); if(e.shiftKey) s.scrollIntoView({block:'center'});}
  });
  refresh();
}

// ---- color heatmap on text ----
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

// ---- time-series chart ----
function sizeCanvas(){ CW=cv.clientWidth; CH=cv.clientHeight;
  cv.width=CW*DPR; cv.height=CH*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }
function drawBase(){
  const sc=D.models[cur.model].word_scores[cur.emotion]; if(!sc) return;
  sizeCanvas(); if(!CW) return; ctx.clearRect(0,0,CW,CH);
  const mid=CH/2, scale=cur.scale;
  ctx.strokeStyle='#2a2f3a'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(CW,mid); ctx.stroke();
  const sl=document.getElementById('showlaugh');
  if(!sl||sl.checked){ ctx.strokeStyle='rgba(226,87,76,0.28)'; ctx.lineWidth=1;
    D.laugh_after_word.forEach(j=>{const x=j/N*CW; ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,CH);ctx.stroke();}); }
  ctx.strokeStyle='#7fb0ff'; ctx.lineWidth=1.4; ctx.beginPath();
  for(let px=0;px<CW;px++){
    const a=Math.floor(px/CW*N), b=Math.max(a+1,Math.floor((px+1)/CW*N));
    let s=0,c=0; for(let k=a;k<b&&k<N;k++){s+=sc[k];c++;}
    const z=c?s/c:0; const y=mid-Math.max(-1,Math.min(1,z/scale))*mid*0.92;
    if(px===0)ctx.moveTo(px,y); else ctx.lineTo(px,y);
  }
  ctx.stroke(); baseImg=ctx.getImageData(0,0,cv.width,cv.height);
}
function drawCursor(wordIdx){
  if(!baseImg) return; ctx.putImageData(baseImg,0,0); if(wordIdx==null) return;
  const x=wordIdx/N*CW; ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,CH); ctx.stroke();
  const sc=D.models[cur.model].word_scores[cur.emotion];
  document.getElementById('chartHover').textContent=`“${D.words[wordIdx]}”  z=${sc?sc[wordIdx].toFixed(2):'?'}  (word ${wordIdx}/${N})`;
}

// ---- probe x scenario figure (SVG, paper-style) ----
function rdbu(t){ // t in [-1,1] -> blue/white/red
  const W=[247,247,247], B=[33,102,172], R=[178,24,43];
  let a=W,b,f; if(t<0){b=B;f=Math.min(1,-t);} else {b=R;f=Math.min(1,t);}
  return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*f)).join(',')})`;
}
function renderHeatmap(){
  const body=document.getElementById('hmbody');
  document.getElementById('hmtitle').textContent=`Emotion Probe × Scenario — ${cur.model}`;
  const sm=D.models[cur.model].scenario_matrix;
  if(!sm||!sm.values){ body.innerHTML='<div style="color:#888;font-size:12px;padding:14px">not computed for this model yet</div>'; return; }
  const rows=sm.rows, cols=sm.cols, V=sm.values;
  let mx=0; V.forEach(r=>r.forEach(v=>{if(Math.abs(v)>mx)mx=Math.abs(v);})); mx=Math.max(mx,1e-3);
  const cell=22, LW=84, TOP=4, GW=cols.length*cell, GH=rows.length*cell, COLH=104, CB=70;
  const W=LW+GW+CB, H=TOP+GH+COLH;
  let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for(let ri=0;ri<rows.length;ri++) for(let cj=0;cj<cols.length;cj++){
    const v=V[ri][cj], x=LW+cj*cell, y=TOP+ri*cell;
    s+=`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${rdbu(v/mx)}" stroke="#fff" stroke-width="0.5">`+
       `<title>${rows[ri]} × ${cols[cj]} = ${v.toFixed(3)}</title></rect>`;
  }
  for(let ri=0;ri<rows.length;ri++)
    s+=`<text x="${LW-5}" y="${TOP+ri*cell+cell/2+3.5}" text-anchor="end" font-size="11">${rows[ri]}</text>`;
  for(let cj=0;cj<cols.length;cj++){
    const x=LW+cj*cell+cell/2, y=TOP+GH+6;
    s+=`<text transform="rotate(-42 ${x} ${y})" x="${x}" y="${y}" text-anchor="end" font-size="10.5">${cols[cj]}</text>`;
  }
  s+=`<text x="${10}" y="${TOP+GH/2}" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90 10 ${TOP+GH/2})">Emotion Probe</text>`;
  const cbx=LW+GW+18, cbw=12, cbh=GH;
  s+=`<defs><linearGradient id="cbg" x1="0" y1="0" x2="0" y2="1">`+
     `<stop offset="0%" stop-color="${rdbu(1)}"/><stop offset="50%" stop-color="${rdbu(0)}"/><stop offset="100%" stop-color="${rdbu(-1)}"/></linearGradient></defs>`;
  s+=`<rect x="${cbx}" y="${TOP}" width="${cbw}" height="${cbh}" fill="url(#cbg)" stroke="#bbb" stroke-width="0.5"/>`;
  [[1,'+'+mx.toFixed(2)],[0.5,'+'+(mx/2).toFixed(2)],[0,'0'],[-0.5,'-'+(mx/2).toFixed(2)],[-1,'-'+mx.toFixed(2)]].forEach(([t,lab])=>{
    const y=TOP+(1-(t+1)/2)*cbh; s+=`<text x="${cbx+cbw+3}" y="${y+3.5}" font-size="9.5">${lab}</text>`;
  });
  s+=`<text x="${cbx+cbw+34}" y="${TOP+cbh/2}" font-size="10" font-weight="600" text-anchor="middle" transform="rotate(90 ${cbx+cbw+34} ${TOP+cbh/2})">Cosine similarity</text>`;
  body.innerHTML=s+'</svg>';
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
window.addEventListener('resize',()=>{ if(viewerReady && document.getElementById('tab-viewer').classList.contains('active')){drawBase();drawCursor(null);} });
</script>
__EPI_SCRIPT__
</body></html>"""

# ---- Epiplexity: in-page tab embedding data/joke/_epiplexity.json -------------
# (produced by build_epiplexity.py — run it before this script). If the artifact
# is absent the tab is silently omitted.
EPI = json.load(open("data/joke/_epiplexity.json")) if os.path.exists("data/joke/_epiplexity.json") else None

EPI_STYLE = """
  :root { --e1:#a07cff; --efull:#7fb0ff; --eshort:#ffb05c; }
  #tab-epi .epi-scroll { height:calc(100vh - 86px); overflow:auto; }
  #tab-epi .epi-inner { max-width:1060px; margin:0 auto; padding:26px 28px 90px; }
  #tab-epi h1 { font-size:25px; margin:0 0 10px; line-height:1.22; }
  #tab-epi h2 { font-size:18px; margin:28px 0 8px; padding-top:10px; border-top:1px solid #20242d; }
  #tab-epi p { margin:10px 0; } #tab-epi .lede { font-size:16px; color:#dfe3ea; }
  #tab-epi a { color:#7fb0ff; }
  #tab-epi code { background:#11141a; border:1px solid #2c313b; border-radius:4px; padding:.5px 5px; font-size:13px; }
  #tab-epi .eq { background:#11141a; border:1px solid #2c313b; border-radius:8px; padding:12px 14px; font-size:14px; color:#dfe3ea; margin:12px 0; overflow-x:auto; }
  #tab-epi table { border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }
  #tab-epi th,#tab-epi td { text-align:left; padding:8px 10px; border-bottom:1px solid #20242d; }
  #tab-epi th { color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  #tab-epi td b { color:#fff; }
  #tab-epi .sig { color:#7ee081; } #tab-epi .ns { color:#ff8a7a; }
  #tab-epi .sw { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:middle; margin-right:5px; }
  #tab-epi .chartcard { background:#0c0e12; border:1px solid #262a33; border-radius:10px; padding:12px 14px; margin:14px 0; }
  #tab-epi .ctrls { display:flex; flex-wrap:wrap; gap:14px; align-items:center; font-size:13px; margin-bottom:8px; }
  #tab-epi .ctrls label { display:flex; align-items:center; gap:6px; cursor:pointer; }
  #tab-epi .ctrls input[type=checkbox]{ accent-color:var(--accent); }
  #tab-epi .chartlbl { font-size:11.5px; color:var(--mut); display:flex; justify-content:space-between; margin:2px 0; }
  #tab-epi #e_chart { width:100%; height:230px; display:block; cursor:crosshair; }
  #tab-epi #e_peri { width:100%; height:230px; display:block; }
  #tab-epi .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:820px){ #tab-epi .grid2{ grid-template-columns:1fr; } }
  #tab-epi .key { font-size:12px; color:var(--mut); }
  #tab-epi blockquote { margin:14px 0; padding:8px 16px; border-left:3px solid var(--accent); background:#14171e; color:#cfd3da; border-radius:0 7px 7px 0; }
"""

EPI_SECTION = """
<section id="tab-epi" class="tab"><div class="epi-scroll"><div class="epi-inner">
<h1>Can epiplexity predict laughter?</h1>
<p class="lede">A first test of the <a href="https://arxiv.org/abs/2601.03220" target="_blank">epiplexity</a>
hypothesis (Finzi et al. 2026): laughter should track the <em>structured, learnable</em> component of
surprise &mdash; not raw surprise. We read <span id="e_hd"></span> token-by-token with a fixed base model
and ask whether the signal rises into the audience's laughs (laughter stripped before the model &mdash; no leakage).</p>
<div class="eq">E1(t) = &minus;log p(x<sub>t</sub> | last K) &minus; (&minus;log p(x<sub>t</sub> | all prior))
= <b style="color:var(--eshort)">s_short</b> &minus; <b style="color:var(--efull)">s_full</b>
= <b style="color:var(--e1)">in-context epiplexity</b>
<div class="key" style="margin-top:6px">nats the long-range setup saves on this token (K = <span id="e_kk"></span>); raw surprisal <code>s_full</code> is the H1 baseline.</div></div>
<h2>Does the signal rise into the laugh?</h2>
<p class="key">Mean z over the <span id="e_pw"></span>-token run-up before each laugh vs a 5,000-sample permutation null; AUC treats &ldquo;within the run-up&rdquo; as the label.</p>
<table id="e_stbl"><thead><tr><th>signal</th><th>run-up z</th><th>&sigma; above null</th><th>p-value</th><th>AUC</th><th>offset 0</th></tr></thead><tbody></tbody></table>
<div class="chartcard">
  <div class="ctrls">
    <label><input type="checkbox" id="e_c_e1" checked><span class="sw" style="background:var(--e1)"></span>E1 epiplexity</label>
    <label><input type="checkbox" id="e_c_full" checked><span class="sw" style="background:var(--efull)"></span>raw surprisal</label>
    <label><input type="checkbox" id="e_c_short"><span class="sw" style="background:var(--eshort)"></span>weak-obs s_short</label>
    <label><input type="checkbox" id="e_c_laugh" checked>laughter ticks</label>
    <label>smooth <input type="range" id="e_sm" min="1" max="81" step="2" value="31"></label>
    <label>view all <input type="checkbox" id="e_c_all" checked></label>
  </div>
  <div class="chartlbl"><span>per-token signal (z-scored, smoothed) across the routine</span><span id="e_hover"></span></div>
  <canvas id="e_chart"></canvas>
  <div class="key">hover to inspect a token &middot; uncheck &ldquo;view all&rdquo; then drag to scan a 700-token window</div>
  <input type="range" id="e_pan" min="0" max="100" value="0" style="width:100%;display:none;margin-top:6px;accent-color:var(--accent)">
</div>
<div class="grid2">
  <div class="chartcard">
    <div class="chartlbl"><span>peri-laughter average (&plusmn;40 tokens, z)</span><span class="key">0 = last word before laugh</span></div>
    <canvas id="e_peri"></canvas>
  </div>
  <div>
    <h2 style="margin-top:0;border:0">What this run shows</h2>
    <p><b>Raw surprisal rises into laughs; forward epiplexity does not</b> &mdash; and that null is the
    theoretically <em>right</em> one. <code>E1</code> asks whether the long setup makes the punchline
    <em>more</em> predictable. For a real joke it doesn't: the punchline is an <b>incongruity</b>, surprising
    <em>given</em> the setup. So E1&asymp;0 at the laugh is the signature of incongruity <em>without forward resolution</em>.</p>
    <blockquote>The &ldquo;getting it&rdquo; &mdash; the resolution that makes incongruity funny &mdash; is
    <b>retrodictive</b>: how much seeing the punchline lets you recompress the <em>setup</em>. A forward
    estimator can't see it. That motivates <b>E1-backward</b> as the next run.</blockquote>
    <p class="key">Sanity check: E1's highest tokens are callbacks/repetitions (e.g. &ldquo;Why do you think the Klan&hellip;&rdquo; recurs 3&times;) &mdash; real structure, just not concentrated pre-laugh.</p>
  </div>
</div>
<p class="key" style="margin-top:24px;padding-top:14px;border-top:1px solid #20242d">Model: <span id="e_md"></span> (base, fixed) &middot; offset-0 spike = punctuation-dominated punchline boundary (a known confound) &middot; single comedian, no orthogonalization yet &mdash; exploratory.</p>
</div></div></section>
"""

EPI_SCRIPT = """<script id="epidata" type="application/json">__EPIDATA__</script>
<script>
window.__epi = (function(){
  const el=document.getElementById('epidata'); if(!el) return null;
  const D=JSON.parse(el.textContent);
  const N=D.n_tokens, laughs=D.laugh_positions;
  const COL={e1:'#a07cff', s_full:'#7fb0ff', s_short:'#ffb05c'};
  const $=id=>document.getElementById(id);
  function zsc(a){let m=0;for(const x of a)m+=x;m/=a.length;let s=0;for(const x of a)s+=(x-m)*(x-m);s=Math.sqrt(s/a.length)||1e-8;return a.map(x=>(x-m)/s);}
  const Z={e1:zsc(D.e1), s_full:zsc(D.s_full), s_short:zsc(D.s_short)};
  function smooth(a,k){if(k<=1)return a.slice();const q=[];let sum=0,h=(k-1)/2,out=new Array(a.length);
    for(let i=0;i<a.length+h;i++){if(i<a.length){sum+=a[i];q.push(a[i]);}if(q.length>k){sum-=q.shift();}const idx=i-h;if(idx>=0)out[idx]=sum/q.length;}return out;}
  let cv,cx,pv,px,CW=0,CH=0,DPR=window.devicePixelRatio||1,SM={},ready=false;
  function sel(){return {e1:$('e_c_e1').checked,s_full:$('e_c_full').checked,s_short:$('e_c_short').checked};}
  function win(){ if($('e_c_all').checked)return[0,N]; const w=700,s=Math.round($('e_pan').value/100*(N-w)); return [Math.max(0,s),Math.min(N,s+w)];}
  function recompute(){const k=+$('e_sm').value;SM={e1:smooth(Z.e1,k),s_full:smooth(Z.s_full,k),s_short:smooth(Z.s_short,k)};}
  function size(){CW=cv.clientWidth;CH=cv.clientHeight;cv.width=CW*DPR;cv.height=CH*DPR;cx.setTransform(DPR,0,0,DPR,0,0);}
  function draw(){ if(!cv)return; size(); if(!CW)return; cx.clearRect(0,0,CW,CH);
    const [a,b]=win(),span=b-a,mid=CH/2,scl=2.6;
    cx.strokeStyle='#2a2f3a';cx.lineWidth=1;cx.beginPath();cx.moveTo(0,mid);cx.lineTo(CW,mid);cx.stroke();
    if($('e_c_laugh').checked){cx.strokeStyle='rgba(226,87,76,0.30)';cx.lineWidth=1;laughs.forEach(j=>{if(j<a||j>=b)return;const x=(j-a)/span*CW;cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,CH);cx.stroke();});}
    const s=sel();
    for(const k of ['s_short','s_full','e1']){ if(!s[k])continue; const arr=SM[k]; cx.strokeStyle=COL[k]; cx.lineWidth=k==='e1'?1.6:1.1; cx.globalAlpha=k==='e1'?1:0.85; cx.beginPath();
      for(let p=0;p<CW;p++){const i0=a+Math.floor(p/CW*span),i1=Math.max(i0+1,a+Math.floor((p+1)/CW*span));let sum=0,c=0;for(let i=i0;i<i1&&i<b;i++){sum+=arr[i];c++;}const v=c?sum/c:0;const y=mid-Math.max(-1,Math.min(1,v/scl))*mid*0.94;p?cx.lineTo(p,y):cx.moveTo(p,y);}
      cx.stroke();cx.globalAlpha=1; }
  }
  function hover(clientX){const r=cv.getBoundingClientRect();const [a,b]=win(),span=b-a;const i=Math.max(a,Math.min(b-1,a+Math.round((clientX-r.left)/CW*span)));
    const near=laughs.some(p=>p>=i&&p-i<15);
    $('e_hover').innerHTML='tok '+i+' \\u201c'+(D.tokens[i]||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')+'\\u201d \\u00b7 <span style=\"color:'+COL.e1+'\">E1 '+(D.e1[i]>=0?'+':'')+D.e1[i]+'</span> \\u00b7 <span style=\"color:'+COL.s_full+'\">surp '+D.s_full[i]+'</span>'+(near?' \\u00b7 <span style=\"color:#ff8a7a\">\\u21b3laugh\\u226415</span>':'');}
  function drawPeri(){ if(!pv)return; const W=pv.clientWidth,H=pv.clientHeight; if(!W)return; pv.width=W*DPR;pv.height=H*DPR;px.setTransform(DPR,0,0,DPR,0,0);px.clearRect(0,0,W,H);
    const off=D.peri.e1.off,n=off.length;let lo=1e9,hi=-1e9;['e1','s_full','s_short'].forEach(k=>D.peri[k].mean.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));const pad=(hi-lo)*0.12||1;lo-=pad;hi+=pad;
    const X=i=>i/(n-1)*(W-44)+34,Y=v=>H-20-(v-lo)/(hi-lo)*(H-34);
    px.strokeStyle='#2a2f3a';px.beginPath();px.moveTo(34,Y(0));px.lineTo(W-10,Y(0));px.stroke();
    const x0=X((n-1)/2);px.strokeStyle='rgba(226,87,76,0.6)';px.lineWidth=1.2;px.beginPath();px.moveTo(x0,4);px.lineTo(x0,H-16);px.stroke();
    px.fillStyle='#9aa0aa';px.font='10px sans-serif';px.fillText('0',x0-3,H-4);px.fillText('-40',30,H-4);px.fillText('+40',W-26,H-4);
    ['s_short','s_full','e1'].forEach(k=>{const m=D.peri[k].mean;px.strokeStyle=COL[k];px.lineWidth=k==='e1'?2:1.3;px.beginPath();m.forEach((v,i)=>{const x=X(i),y=Y(v);i?px.lineTo(x,y):px.moveTo(x,y);});px.stroke();});}
  function buildTable(){ const tb=document.querySelector('#e_stbl tbody'); if(!tb||tb.childElementCount)return;
    [['e1','E1 epiplexity',COL.e1],['s_full','raw surprisal s_full',COL.s_full],['s_short','weak-obs s_short',COL.s_short]].forEach(function(row){var k=row[0],lbl=row[1],c=row[2];var s=D.stats[k];var sc=s.p<0.05?'sig':'ns';var tr=document.createElement('tr');
      tr.innerHTML='<td><span class=\"sw\" style=\"background:'+c+'\"></span>'+lbl+'</td><td>'+(s.run_z>=0?'+':'')+s.run_z.toFixed(3)+'</td><td>'+(s.sigma>=0?'+':'')+s.sigma.toFixed(1)+'\\u03c3</td><td class=\"'+sc+'\"><b>'+s.p.toFixed(4)+'</b>'+(s.p<0.05?'':' n.s.')+'</td><td>'+s.auc.toFixed(3)+'</td><td>'+(s.off0_z>=0?'+':'')+s.off0_z.toFixed(2)+'</td>';
      tb.appendChild(tr);}); }
  function init(){ if(ready)return; ready=true;
    cv=$('e_chart');cx=cv.getContext('2d');pv=$('e_peri');px=pv.getContext('2d');
    $('e_hd').textContent=D.name+' ('+N.toLocaleString()+' tokens, '+D.n_laughs+' laughs)';
    $('e_kk').textContent=D.short_ctx;$('e_pw').textContent=D.pre_window;$('e_md').textContent=D.model;
    buildTable();recompute();
    cv.addEventListener('mousemove',function(e){hover(e.clientX);});
    ['e_c_e1','e_c_full','e_c_short','e_c_laugh'].forEach(function(id){$(id).onchange=draw;});
    $('e_sm').oninput=function(){recompute();draw();};
    $('e_c_all').onchange=function(){$('e_pan').style.display=$('e_c_all').checked?'none':'block';draw();};
    $('e_pan').oninput=draw;
    window.addEventListener('resize',function(){if(document.getElementById('tab-epi').classList.contains('active')){draw();drawPeri();}});
  }
  function redraw(){recompute();draw();drawPeri();}
  return {init:init, redraw:redraw};
})();
</script>"""

EPI_NAV = '<button data-tab="epi">Epiplexity</button>' if EPI else ''
EPI_SCRIPT_FULL = EPI_SCRIPT.replace("__EPIDATA__", json.dumps(EPI)) if EPI else ''

out = (HTML.replace("__DATA__", json.dumps(DATA))
           .replace("__INTRO__", INTRO_HTML)
           .replace("__LIT__", LIT_HTML)
           .replace("__EPI_STYLE__", EPI_STYLE if EPI else "")
           .replace("__EPI_NAV__", EPI_NAV)
           .replace("__EPI_SECTION__", EPI_SECTION if EPI else "")
           .replace("__EPI_SCRIPT__", EPI_SCRIPT_FULL))
open("index.html", "w").write(out)
open("viewer.html", "w").write(out)
print(f"wrote index.html + viewer.html ({len(out)/1e6:.1f} MB) — {len(models)} models, "
      f"{len(DATA['words'])} words, {len(DATA['emotions'])} emotions, "
      f"lit {len(LIT_HTML)//1000}KB; "
      f"matrices: {[t for t,m in models.items() if m.get('scenario_matrix')]}")
