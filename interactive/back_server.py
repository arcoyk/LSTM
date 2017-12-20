from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request
from urllib.parse import urlparse
import json
import brain

class Handler(SimpleHTTPRequestHandler):
  def do_POST(self):
    self.data_string = self.rfile.read(int(self.headers['Content-Length']))
    self.send_response(200)
    self.end_headers()
    data = self.data_string.decode('utf-8')
    rst = brain.learn_and_answer(data)
    rst = json.dumps(rst)
    body = bytes(rst, 'utf-8')
    self.wfile.write(body)
    print('returning', rst)
    return

HOST = 'localhost'
PORT = 8000
httpd = HTTPServer((HOST, PORT), Handler)
print('Backend brain alive 🔥 ', HOST, ':', PORT)
httpd.serve_forever()
