import subprocess
import os
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Start Streamlit only if not running
        port = os.environ.get("PORT", "8501")
        process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port", port, "--server.headless", "true", "--server.enableCORS", "false"]
        )
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        message = f"<h2>🚀 Streamlit app starting...</h2><p>Once ready, open: <a href='https://{os.environ.get('VERCEL_URL')}' target='_blank'>https://{os.environ.get('VERCEL_URL')}</a></p>"
        self.wfile.write(message.encode())
        return
