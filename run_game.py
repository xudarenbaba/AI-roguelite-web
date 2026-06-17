"""Serve the game client as static files (development use)."""
from __future__ import annotations

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "game"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[game] {self.address_string()} - {fmt % args}")


print(f"Game client : http://127.0.0.1:{PORT}")
print(f"NPC API     : http://127.0.0.1:5100  (run `python run.py` separately)")
print("Press Ctrl+C to stop.\n")

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
