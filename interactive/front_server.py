import requests
import os
HOST = 'localhost'
PORT = 9000
IN_FILE = 'input.bytes'
OUT_FILE = 'output.bytes'

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
    in_data = f.read().split('\n')[-1]
  try:
    print('> ', in_data)
    res = requests.post(SERVER, data=in_data)
    with open(OUT_FILE, 'w') as f:
      print('< ', res.text)
      f.write(res.text)
  except:
    print('Backend server not responding')

print("Frontend sockets alive ⚡️")
print("Expecting backend ", HOST, ':', PORT)
while True:
  if in_file_updated():
    update_out_file()
