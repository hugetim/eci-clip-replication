"""The clip post's benchmark-by-benchmark chart of two models (Plotly, inlined so the page works offline).

Each row is a benchmark at least one of the two models was scored on, hardest (highest ECI at a 50% score) at the top.
The line is the benchmark's scale from a 0.1% to a 99.9% score (Epoch's clip), ticks at 10%, ..., 90%, every point
placed at the ECI that score implies under Epoch's beta likelihood (where the score is most likely; scripts/
eci_weights_beta.py). Dots: the two models' scores on that scale; numbers: the scores (the higher in bold); dashed lines:
the models' published ECIs. Benchmark names link to Epoch's benchmark pages; a dagger marks benchmarks whose scores
Epoch rescales (chance floor, ceiling).

Run in the folder holding scripts/eci_weights_beta.py's outputs (eci_benchmark_weights.csv, benchmark_curves_eci.csv).
Usage: ECI_DATA_ROOT=<unpacked Epoch archive> python viz/pair_panel.py "Model A|Model B" "Label A|Label B" OUT.html
Also writes <OUT>_curves.csv (each shown benchmark's curve) and <OUT>_scores.csv (the two models' scores, with Epoch's
published ECI and the ECI each score implies).
"""
import json, sys, os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, expit, logit
sys.path.insert(0, os.path.dirname(__file__))
from _style import HEAD

HERE = os.path.dirname(os.path.abspath(__file__))
PAIR, LABELS, OUT = sys.argv[1].split("|"), sys.argv[2].split("|"), sys.argv[3]
w = pd.read_csv("eci_benchmark_weights.csv", parse_dates=["date"])
bc = pd.read_csv("benchmark_curves_eci.csv").set_index("benchmark")
DATA = os.environ["ECI_DATA_ROOT"]  # unpacked Epoch archive
ci = pd.read_csv(os.path.join(DATA, "epoch_capabilities_index", "eci_scores.csv")).set_index("Model")
meta = pd.read_csv(os.path.join(DATA, "benchmark_metadata.csv")).set_index("benchmark")
pages = pd.read_csv(os.path.join(HERE, "epoch_benchmark_pages.csv")).set_index("benchmark")
assert len(PAIR) == 2 and all(x in set(w.model) for x in PAIR), f"need two models in the weights file, got {PAIR}"
# How far below the frontier (the best ECI reached by its release date, from GPT-4 on; same-day releases count) each
# model was at release, for the header
m = w.groupby("model").agg(ECI=("ECI", "first"), date=("date", "first")).reset_index()
m = m[m.date >= "2023-03-14"].sort_values(["date", "ECI"], ascending=[True, False])
frontier = set(m[m.ECI > m.ECI.shift(1).cummax().fillna(-np.inf)].model)
gap = dict(zip(m.model, m.ECI.cummax() - m.ECI))


def implied_eci(p, D, a, k):
    """ECI at which a score p is most likely under the beta likelihood (chat in scripts/eci_weights_beta.py)."""
    def score(C):
        mu = np.clip(expit(a * (C - D)), 1e-6, 1 - 1e-6)
        return logit(p) - digamma(mu * k) + digamma((1 - mu) * k)
    return brentq(score, -2000.0, 2000.0)


# Benchmark info: display name, Epoch page, curve, Epoch's score adjustment (chance floor, ceiling), scale ticks
RULER_P = [0.001] + [q / 10 for q in range(1, 10)] + [0.999]  # the scale's ends (Epoch's clip) and its ticks
bench = {}
for b in sorted(set(w[w.model.isin(PAIR)].benchmark)):
    rb = float(meta.loc[b, "random_baseline"]) if b in meta.index and pd.notna(meta.loc[b, "random_baseline"]) else 0.0
    c = float(meta.loc[b, "score_ceiling"]) if b in meta.index and pd.notna(meta.loc[b, "score_ceiling"]) else 1.0
    rb = rb if rb >= 0.01 else 0.0  # OTIS's 0.1% baseline changes nothing visible
    pg = pages.loc[b] if b in pages.index else None
    bench[b] = {"name": pg.display if pg is not None else b,
                "short": pg.short if pg is not None and isinstance(pg.short, str) else (pg.display if pg is not None else b),
                "url": f"https://epoch.ai/benchmarks/{pg.slug}" if pg is not None else "https://epoch.ai/benchmarks/search",
                "D": round(bc.loc[b, "mid_eci"], 2), "rb": round(rb, 4), "c": round(c, 4),
                "t": [round(implied_eci(p, bc.loc[b, "mid_eci"], bc.loc[b, "alpha_eci"], bc.loc[b, "kappa"]), 3) for p in RULER_P]}
missing = [b for b in bench if b not in pages.index]
if missing:
    print("WARNING: no Epoch page mapped for:", missing, "(add them to viz/epoch_benchmark_pages.csv)")

data = {}
for name in PAIR:
    g = w[w.model == name]
    rows = [[r.benchmark, round(r.performance, 4), round(r.implied_ECI, 3)] for r in g.itertuples()]
    c = ci.loc[name]
    data[name] = {"eci": round(g.ECI.iloc[0], 1), "date": str(g.date.iloc[0].date()),
                  "lo": None if pd.isna(c.eci_ci_low) else round(c.eci_ci_low, 1),
                  "hi": None if pd.isna(c.eci_ci_high) else round(c.eci_ci_high, 1),
                  "neff": round(1 / (g.share ** 2).sum(), 1), "rows": rows,
                  "fr": name in frontier, "gap": round(gap.get(name, 0.0), 1)}
_info = os.path.join(DATA, "ARCHIVE_INFO.json")
DATA_DATE = json.load(open(_info)).get("data_date", "") if os.path.exists(_info) else ""
PLOTLY = open(os.path.join(HERE, "vendor", "plotly-basic-2.35.2.min.js"), encoding="utf-8").read()

note = ("Each row is a benchmark, with both models' scores on it, placed at the ECI each score implies under Epoch's "
        "model, so that every benchmark is on the same ECI scale. Hardest benchmarks (highest ECI at a 50% score) at the top.")
html = HEAD.format(title="Compare what each model's ECI rests on", note=note) + """
<style>
.hdr{font-size:14px;font-weight:500;margin:8px 0 0}.sub{font-size:11.5px;color:var(--text-muted);margin:0 0 2px}
#plot a{fill:inherit}
.foot{font-size:12px;color:var(--text-muted);margin:4px 0}
</style>
<div class="legend" id="legend"></div>
<div id="hdr"></div>
<div id="plot"></div>
<p id="info" style="font-size:13px;color:var(--text-secondary);min-height:40px">Hover over or tap a dot for the score behind it; click a benchmark name for Epoch's page on it.</p>
<p class="foot">&dagger; Epoch rescales this benchmark's scores before fitting: random guessing counts as 0% and, for some, a maximum below 100% counts as 100%. Scores here are the rescaled ones; hover a dot for the raw score. Data: <a href="https://epoch.ai/benchmarks" target="_blank" style="color:inherit">Epoch AI Benchmarking Hub</a>, ECI as published__ASOF__.</p>
<script>__PLOTLY__</script>
<script>
const DATA=__D__,B=__B__,PAIR=__PAIR__,LABELS=__LABELS__;
let X0=100,X1=200;/* ECI axis range, set in render() to what the two models need */
const COL=['#D85A30','#5B57D1'],GREEN='#1D9E75',DARKG='#085041';
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim()||'#52514e';
const pct=x=>(Math.round(x*1000)/10)+'%';
const adj=b=>b.rb>0||b.c<1;
/* a score's dot sits at the ECI it implies (v[2]); the scale's ticks likewise (b.t: 0.1%, 10%, ..., 90%, 99.9%) */
const dotE=(b,v)=>v[2];
const tickE=(b,q)=>b.t[q];/* q = 0..10 */
const info=document.getElementById('info');
function scoreText(b,p){if(!adj(b))return `scored ${pct(p)}`;
 const raw=b.rb+p*(b.c-b.rb);let s=`Epoch-adjusted score ${pct(p)} (raw ${p<=0.0011?'at or below '+pct(b.rb):p>=0.9989&&b.c<1?'at or above '+pct(b.c):pct(raw)}`;
 const parts=[];if(b.rb>0)parts.push(`random guessing ${pct(b.rb)} counts as 0%`);if(b.c<1)parts.push(`${pct(b.c)} counts as 100%`);return s+'; '+parts.join(', ')+')';}
function dotText(n,bn,v){const b=B[bn],p=v[1],e=dotE(b,v);
 return `<b>${n}</b> \\u00b7 ${b.name}<br>${scoreText(b,p)}<br>implies ECI ${e.toFixed(1)}${p<=0.0011||p>=0.9989?' (as clipped by Epoch)':''}`;}
function header(shown){document.getElementById('hdr').innerHTML=shown.map((n,k)=>{const m=DATA[n],ciT=m.lo?` (90% CI ${m.lo}\\u2013${m.hi})`:'';
 return `<p class="hdr" style="color:${COL[k]}">${n} \\u00b7 ECI ${m.eci.toFixed(1)}${ciT}</p><p class="sub">released ${m.date} \\u00b7 ${m.rows.length} benchmarks \\u00b7 effective number ${m.neff}${m.fr?'':' \\u00b7 '+m.gap.toFixed(1)+' below the frontier at release'}</p>`;}).join('');
 const sw2=(c,l)=>`<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${c}"/></svg><svg width="10" height="12"><line x1="5" y1="0" x2="5" y2="12" stroke="${c}" stroke-width="1.5" stroke-dasharray="3 2"/></svg>${l}</span>`;
 document.getElementById('legend').innerHTML=`<span><svg width="30" height="12"><line x1="1" y1="6" x2="29" y2="6" stroke="${GREEN}"/><line x1="8" y1="3" x2="8" y2="9" stroke="${DARKG}"/><line x1="15" y1="1" x2="15" y2="11" stroke="${DARKG}"/><line x1="22" y1="3" x2="22" y2="9" stroke="${DARKG}"/></svg>Benchmark scale: the ECI each score implies, from 0.1% to 99.9% (Epoch\\u2019s clip), with ticks at 10%, 20%, \\u2026, 90% (long tick: 50%)</span>`+shown.map((n,k)=>sw2(COL[k],n+': score on the scale (dot), ECI (dashed)')).join('');}
function render(shown){const ms=shown.map(n=>DATA[n]);
 {/* ECI axis: just the span of the two models' dots and ECIs, within 30 points of their ECIs; padded, on a 5-point grid */
  const mid=ms.reduce((a,m)=>a+m.eci,0)/ms.length,v=[...ms.map(m=>m.eci),...ms.flatMap(m=>m.rows.map(r=>dotE(B[r[0]],r)))];
  const lo=Math.max(Math.min(...v),mid-30),hi=Math.min(Math.max(...v),mid+30);
  X0=Math.floor((lo-2)/5)*5;X1=Math.ceil((hi+2)/5)*5;}
 const byB={};ms.forEach((m,k)=>m.rows.forEach(r=>{(byB[r[0]]=byB[r[0]]||[null,null])[k]=r;}));
 const rows=Object.keys(byB).sort((a,b)=>B[a].D-B[b].D);/* y counts up, so the hardest benchmark ends up at the top */
 const n=rows.length,W=document.getElementById('plot').clientWidth,narrow=W<700,txt=css('--text-secondary'),mut=css('--text-muted'),grid=css('--border');
 const lx=[],ly=[],tx=[],ty=[],hx=[],hy=[],ht=[];
 rows.forEach((bn,i)=>{const b=B[bn];
  const c0=tickE(b,0),c9=tickE(b,10);lx.push(c0,c9,null);ly.push(i,i,null);/* the scale runs to Epoch's clip, 0.1%-99.9%, with no end caps (Tim's choice) */
  for(let q=1;q<=9;q++){const e=tickE(b,q),t=q===5?.22:.11;tx.push(e,e,null);ty.push(i-t,i+t,null);}
  hx.push(b.D);hy.push(i);ht.push(`<b>${b.name}</b>${adj(b)?' \\u2020':''}<br>a 50% score implies ECI ${b.D.toFixed(0)}; 10% and 90% imply ${b.t[1].toFixed(0)} and ${b.t[9].toFixed(0)}`);});
 const tr=[{x:lx,y:ly,type:'scatter',mode:'lines',line:{color:GREEN,width:1},opacity:.6,hoverinfo:'skip'},
  {x:tx,y:ty,type:'scatter',mode:'lines',line:{color:DARKG,width:1},hoverinfo:'skip'},
  {x:hx,y:hy,type:'scatter',mode:'markers',marker:{size:14,opacity:0},text:ht,hovertemplate:'%{text}<extra></extra>'}];
 const off=[.14,-.14];
 ms.forEach((m,k)=>{const X=[],Y=[],S=[],T=[];
  rows.forEach((bn,i)=>{const v=byB[bn][k];if(!v)return;const b=B[bn],e=dotE(b,v),t=dotText(shown[k],bn,v);
   X.push(Math.min(X1-.8,Math.max(X0+.8,e)));Y.push(i+off[k]);S.push(e<X0?'triangle-left':e>X1?'triangle-right':'circle');T.push(t);});
  tr.push({x:X,y:Y,type:'scatter',mode:'markers',marker:{color:COL[k],size:9,symbol:S,line:{color:css('--surface-2')||'#fff',width:1.2}},text:T,hovertemplate:'%{text}<extra></extra>',name:shown[k]});
  const SX=[],SY=[],ST=[],SH=[];rows.forEach((bn,i)=>{const v=byB[bn][k];if(!v)return;const o=byB[bn][1-k],hiK=o&&v[1]>o[1];
   SX.push(k);SY.push(i);ST.push((hiK?'<b>':'')+pct(v[1])+(hiK?'</b>':''));SH.push(dotText(shown[k],bn,v));});
  tr.push({x:SX,y:SY,xaxis:'x4',type:'scatter',mode:'text',text:ST,textfont:{color:COL[k],size:narrow?11:12.5},hovertext:SH,hovertemplate:'%{hovertext}<extra></extra>'});});
 const shapes=ms.map((m,k)=>({type:'line',xref:'x',yref:'y',x0:m.eci,x1:m.eci,y0:-.6,y1:n-.4,line:{color:COL[k],width:1.5,dash:'dot'}}));
 const tick=rows.map(bn=>{const b=B[bn];return `<a href="${b.url}" target="_blank">${narrow?b.short:b.name}${adj(b)?' \\u2020':''}</a>`;});
 const lay={height:n*24+86,margin:{l:narrow?96:200,r:narrow?22:10,t:26,b:40},showlegend:false,hovermode:'closest',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'system-ui,sans-serif',size:narrow?10.5:11.5,color:txt},shapes,dragmode:'zoom',
  xaxis:{domain:narrow?[.38,1]:[.25,1],range:[X0,X1],dtick:X1-X0<=30?5:10,gridcolor:grid,zeroline:false,title:{text:'ECI',font:{size:11,color:mut}},tickfont:{color:mut}},
  xaxis4:{domain:narrow?[0,.36]:[0,.22],range:[-.65,1.65],visible:false,fixedrange:true},
  annotations:shown.map((nm,k)=>({xref:'x4',yref:'paper',x:k,y:1,yanchor:'bottom',showarrow:false,text:'<b>'+(LABELS[k]||nm)+'</b>',font:{color:COL[k],size:narrow?10.5:11.5}})),
  yaxis:{tickvals:rows.map((_,i)=>i),ticktext:tick,range:[-.7,n-.3],showgrid:false,zeroline:false,fixedrange:true,ticks:'',anchor:'free',position:0},
  hoverlabel:{align:'left',font:{size:12}}};
 Plotly.newPlot('plot',tr,lay,{displaylogo:false,responsive:true,modeBarButtons:[['zoom2d','pan2d','resetScale2d']]});
 const pl=document.getElementById('plot');pl.removeAllListeners&&pl.removeAllListeners('plotly_click');
 pl.on('plotly_click',ev=>{const p=ev.points[0];const t=p.hovertext||p.text;if(t)info.innerHTML=t;});}
function renderAll(){header(PAIR);render(PAIR);
 info.textContent='Hover over or tap a dot for the score behind it; click a benchmark name for Epoch\\u2019s page on it.';}
renderAll();
let lastNarrow=null,rt=null;new ResizeObserver(()=>{clearTimeout(rt);rt=setTimeout(()=>{const nw=document.getElementById('plot').clientWidth<700;if(lastNarrow!==null&&nw!==lastNarrow)renderAll();lastNarrow=nw;},150);}).observe(document.getElementById('plot').parentElement);
window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',renderAll);

// tell an embedding page how tall this chart is (works across origins)
if(window.parent!==window){document.documentElement.style.cssText='color-scheme:light;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;--surface-1:#f3f2ee;--surface-2:#fff;--border:rgba(11,11,11,.10)';document.body.style.background='transparent';document.querySelectorAll('body>h1,body>h1+p').forEach(e=>e.style.display='none');document.body.style.margin='0 auto';renderAll();const post=()=>parent.postMessage({eciFrameHeight:Math.ceil(document.body.getBoundingClientRect().height)+24},'*');new ResizeObserver(post).observe(document.body);window.addEventListener('load',post);post();}
</script></body></html>"""
for k, v in {"__D__": json.dumps(data), "__B__": json.dumps(bench), "__PAIR__": json.dumps(PAIR), "__LABELS__": json.dumps(LABELS),
             "__ASOF__": f" (data as of {pd.Timestamp(DATA_DATE):%b %-d, %Y})" if DATA_DATE else ""}.items():
    html = html.replace(k, v)
html = html.replace("<script>__PLOTLY__</script>", "<script>" + PLOTLY.replace("</script", "<\\/script") + "</script>")
open(OUT, "w", encoding="utf-8").write(html)
# the curves behind the chart, for the post's own figure (<OUT>_curves.csv)
pd.DataFrame([{"benchmark": b, "display": v["name"], "mid_eci": bc.loc[b, "mid_eci"], "alpha_eci": bc.loc[b, "alpha_eci"],
               "kappa": bc.loc[b, "kappa"]} for b, v in bench.items()]).to_csv(os.path.splitext(OUT)[0] + "_curves.csv", index=False)
# and the two models' scores, each with Epoch's published ECI and the ECI the score implies (<OUT>_scores.csv)
sc = w[w.model.isin(PAIR)][["model", "ECI", "benchmark", "performance", "implied_ECI"]].copy()
sc.insert(3, "display", sc.benchmark.map(lambda b: bench[b]["name"])); sc.insert(4, "short", sc.benchmark.map(lambda b: bench[b]["short"]))
sc.to_csv(os.path.splitext(OUT)[0] + "_scores.csv", index=False)
print(f"Wrote {OUT}: {PAIR[0]} vs {PAIR[1]}")
