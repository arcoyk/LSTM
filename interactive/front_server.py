import requests
import os
import time

HOST = 'http://localhost'
PORT = '8000'
BACKEND = HOST + ':' + PORT
IN_FILE = 'input.txt'
OUT_FILE = 'output.txt'
INTERVAL = 1

prev = 0
def in_file_updated():
  global prev
  curr = os.path.getmtime(IN_FILE)
  if prev != curr:
    prev = curr
    return True
  return False

def update_out_file():
  in_data = ''
  with open(IN_FILE) as f:
    # Last line
    in_data = f.readlines()[-1]
  print('> ', in_data)
  res = requests.post(BACKEND, data=in_data)
  with open(OUT_FILE, 'a') as f:
    print('< ', res.text)
    f.write(res.text + '\n')

print("Frontend sockets alive ⚡️")
print("Expecting backend ", BACKEND)
while True:
  time.sleep(INTERVAL)
  if in_file_updated():
    update_out_file()
