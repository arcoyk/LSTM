from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request
from urllib.parse import urlparse
import json

class Handler(SimpleHTTPRequestHandler):
  def do_GET(self):
    parsed = urlparse(self.path)
    queries = urllib.parse.parse_qs(parsed.query)
    # method = queries['method'][0]
    # print('method=', method)
    rst = {'hoge': 32}
    rst = json.dumps(rst)
    body = bytes(rst, 'utf-8') 
    self.send_response(200)
    self.send_header('Content-type', 'text/html;charset=utf-8')
    self.send_header('Content-length', len(body))
    self.end_headers()
    self.wfile.write(body)
    
host = 'localhost'
port = 8000
httpd = HTTPServer((host, port), Handler)
print('serving@%d 🎾' % port)
httpd.serve_forever()

