#!/usr/bin/env python3
"""Simple web chat server for CHIMBISIAI - proxies to Ollama"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json, urllib.request, os

os.chdir('/root/chimbisiai')

class ChatHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/generate':
            length = int(self.headers['Content-Length'])
            body = self.rfile.read(length)
            # Proxy to Ollama
            req = urllib.request.Request(
                'http://localhost:11434/api/generate',
                data=body,
                headers={'Content-Type': 'application/json'}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = resp.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(result)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/web-chat.html'
        super().do_GET()

    def log_message(self, format, *args):
        pass  # silence logs

print('CHIMBISIAI Chat Server on port 8080')
HTTPServer(('0.0.0.0', 8080), ChatHandler).serve_forever()
