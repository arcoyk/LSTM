from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request
from urllib.parse import urlparse
import json

def last_line(file_name):
  with open(file_name) as f:
    return f.read().split('\n')[-1]

def push_line(file_name, s):
  with open(file_name, 'a') as f:
    f.write(s)

class Handler(SimpleHTTPRequestHandler):
  def do_POST(self):
    self.data_string = self.rfile.read(int(self.headers['Content-Length']))
    self.data_string = self.data_string.decode('utf-8')
    push_line(INPUT_FILE, self.data_string)
    self.send_response(200)
    self.end_headers()
    rst = last_line(OUTPUT_FILE)
    rst = json.dumps(rst)
    body = bytes(rst, 'utf-8')
    self.wfile.write(body)
    return

INPUT_FILE = 'server_input.bytes'
OUTPUT_FILE = 'server_output.bytes'
host = 'localhost'
port = 8000
httpd = HTTPServer((host, port), Handler)
print('serving@%d' % port)
httpd.serve_forever()
