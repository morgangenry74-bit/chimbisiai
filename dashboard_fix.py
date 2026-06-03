#!/usr/bin/env python3
import subprocess, json, os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

TARGET = 2000
DATA = "/root/chimbisiai/data/train_v3.jsonl"

def cmd(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except:
        return ""

def status():
    samples = cmd(f"wc -l < {DATA} 2>/dev/null || echo 0")
    gen = cmd("ps aux | grep \"python3 -u scripts/generate_v2.py\" | grep -v grep | wc -l")
    trn = cmd("ps aux | grep \"python3.*scripts/train_v3.py\" | grep -v grep | wc -l")
    wlog = cmd("tail -5 /root/chimbisiai/watch_log.txt 2>/dev/null")
    tlog = cmd("tail -15 /root/chimbisiai/train_v3_log.txt 2>/dev/null")
    glog = cmd("tail -3 /root/chimbisiai/gen_v2_log.txt 2>/dev/null")
    try: s = int(samples)
    except: s = 0
    if int(trn or "0") > 0: phase = "TRAINING"
    elif s >= TARGET: phase = "PREPARING"
    elif int(gen or "0") > 0: phase = "GENERATING"
    else: phase = "IDLE"
    return {"samples":s,"target":TARGET,"percent":round(s/TARGET*100,1),"phase":phase,
            "gen_running":int(gen or "0")>0,"train_running":int(trn or "0")>0,
            "watch_log":wlog,"train_log":tlog,"gen_log":glog,
            "updated":datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}

HTML = open("/root/chimbisiai/dashboard.html").read()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/api/status":
            d=status()
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(json.dumps(d).encode())
        else:
            self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.end_headers()
            self.wfile.write(HTML.encode())
    def log_message(self,*a):pass

if __name__=="__main__":
    HTTPServer(("0.0.0.0",8855),H).serve_forever()
