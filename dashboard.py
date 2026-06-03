#!/usr/bin/env python3
"""CHIMBISIAI Training Dashboard — live monitoring (local)"""
import subprocess
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

TARGET_SAMPLES = 2000
DATA_PATH = "/root/chimbisiai/data/train_v3.jsonl"

def local_cmd(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except:
        return ""

def get_status():
    samples = local_cmd(f"wc -l < {DATA_PATH} 2>/dev/null || echo 0")
    gen_running = local_cmd("pgrep -f "python3.*generate_v2" | wc -l")
    train_running = local_cmd("pgrep -f "python3.*train_v3" | wc -l")
    watch_log = local_cmd("tail -5 /root/chimbisiai/watch_log.txt 2>/dev/null")
    train_log = local_cmd("tail -15 /root/chimbisiai/train_v3_log.txt 2>/dev/null")
    gen_log = local_cmd("tail -3 /root/chimbisiai/gen_v2_log.txt 2>/dev/null")
    
    try:
        samples_int = int(samples)
    except:
        samples_int = 0
    
    if int(train_running or "0") > 0:
        phase = "TRAINING"
    elif samples_int >= TARGET_SAMPLES:
        phase = "PREPARING"
    elif int(gen_running or "0") > 0:
        phase = "GENERATING"
    else:
        phase = "UNKNOWN"
    
    return {
        "samples": samples_int,
        "target": TARGET_SAMPLES,
        "percent": round(samples_int / TARGET_SAMPLES * 100, 1),
        "phase": phase,
        "gen_running": int(gen_running or "0") > 0,
        "train_running": int(train_running or "0") > 0,
        "watch_log": watch_log,
        "train_log": train_log,
        "gen_log": gen_log,
        "updated": datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    }

HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHIMBISIAI v3 Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#e0e0e0;font-family:JetBrains Mono,Fira Code,monospace;min-height:100vh;padding:20px}
.c{max-width:900px;margin:0 auto}
h1{text-align:center;color:#22d3ee;font-size:1.8em;margin-bottom:20px;text-shadow:0 0 20px rgba(34,211,238,.3)}
.badge{text-align:center;font-size:1.4em;padding:12px;margin-bottom:20px;background:rgba(34,211,238,.1);border:1px solid #22d3ee;border-radius:8px}
.pbar-bg{width:100%;height:40px;background:#1a1a2e;border-radius:8px;overflow:hidden;border:1px solid #444;position:relative;margin-bottom:15px}
.pbar{height:100%;background:linear-gradient(90deg,#22d3ee,#06b6d4);transition:width .5s;border-radius:8px;box-shadow:0 0 15px rgba(34,211,238,.4)}
.ptxt{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-weight:bold;font-size:1.1em;color:#fff;text-shadow:0 0 5px rgba(0,0,0,.8)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.stat{background:#111;padding:12px;border-radius:8px;text-align:center;border:1px solid #333}
.sv{font-size:1.5em;color:#22d3ee;font-weight:bold}
.sl{font-size:.8em;color:#888;margin-top:4px}
.log{background:#0d0d1a;border:1px solid #333;border-radius:8px;padding:15px;margin-bottom:15px}
.lt{color:#22d3ee;margin-bottom:8px;font-size:.9em}
.lc{font-size:.78em;color:#aaa;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;line-height:1.5}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dg{background:#22c55e;box-shadow:0 0 8px #22c55e}
.dy{background:#eab308;box-shadow:0 0 8px #eab308}
.dr{background:#ef4444;box-shadow:0 0 8px #ef4444}
.ind{display:flex;gap:20px;justify-content:center;margin-bottom:15px}
.upd{text-align:center;color:#555;font-size:.8em;margin-top:15px}
</style>
</head>
<body>
<div class="c">
<h1>CHIMBISIAI v3</h1>
<div class="badge" id="phase">...</div>
<div class="ind">
<div><span class="dot" id="gd"></span>Generator</div>
<div><span class="dot" id="td"></span>Training</div>
</div>
<div class="pbar-bg"><div class="pbar" id="pb" style="width:0%"></div><div class="ptxt" id="pt">0/2000</div></div>
<div class="stats">
<div class="stat"><div class="sv" id="s">-</div><div class="sl">Samples</div></div>
<div class="stat"><div class="sv" id="p">-</div><div class="sl">Progress</div></div>
<div class="stat"><div class="sv" id="eta">-</div><div class="sl">ETA</div></div>
</div>
<div class="log"><div class="lt">Generation Log</div><div class="lc" id="gl">-</div></div>
<div class="log"><div class="lt">Training Log</div><div class="lc" id="tl">Waiting...</div></div>
<div class="log"><div class="lt">Watch Monitor</div><div class="lc" id="wl">-</div></div>
<div class="upd" id="upd">-</div>
</div>
<script>
let ps=0,pt=0;
const phases={"GENERATING":"📝 Генерация данных","TRAINING":"🔥 ОБУЧЕНИЕ","PREPARING":"⏳ Подготовка","UNKNOWN":"❓"};
async function r(){try{const d=await(await fetch("/api/status")).json();
document.getElementById("phase").textContent=phases[d.phase]||d.phase;
document.getElementById("pb").style.width=d.percent+"%";
document.getElementById("pt").textContent=d.samples+"/"+d.target;
document.getElementById("s").textContent=d.samples;
document.getElementById("p").textContent=d.percent+"%";
const n=Date.now();if(ps>0&&d.samples>ps){const rate=(d.samples-ps)/((n-pt)/60000),rem=d.target-d.samples,m=Math.round(rem/rate);document.getElementById("eta").textContent=m>60?Math.floor(m/60)+"h "+m%60+"m":m+"m"}
ps=d.samples;pt=n;
document.getElementById("gd").className="dot "+(d.gen_running?"dg":"dr");
document.getElementById("td").className="dot "+(d.train_running?"dg":"dy");
document.getElementById("gl").textContent=d.gen_log||"-";
document.getElementById("tl").textContent=d.train_log||"Waiting for generation...";
document.getElementById("wl").textContent=d.watch_log||"-";
document.getElementById("upd").textContent="Updated: "+d.updated;
}catch(e){document.getElementById("upd").textContent="Error: "+e.message}}
r();setInterval(r,15000);
</script>
</body></html>"""

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/api/status":
            d=get_status()
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(json.dumps(d).encode())
        else:
            self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.end_headers()
            self.wfile.write(HTML.encode())
    def log_message(self,*a):pass

if __name__=="__main__":
    port=8855
    print(f"Dashboard: http://0.0.0.0:{port}")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
