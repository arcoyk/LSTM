import requests
import os
SERVER = 'http://localhost:8000'
IN_FILE = 'input.bytes'
OUT_FILE = 'out.bytes'

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
  print('POSTing...')
  res = requests.post(SERVER, data=in_data)
  with open(OUT_FILE, 'w') as f:
    print('Saved response:', res.text)
    f.write(res.text)

print("Front file alive ⚡️")
while True:
  if in_file_updated():
    update_out_file()
