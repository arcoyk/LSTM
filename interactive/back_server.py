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

host = 'localhost'
port = 8000
httpd = HTTPServer((host, port), Handler)
print('braing to network interface working@%d' % port)
httpd.serve_forever()
