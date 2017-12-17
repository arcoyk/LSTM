import random

def new_line():
  input = [str(random.random()) for i in range(2)]
  target = [str(random.random()) for i in range(2)]
  return ','.join(input) + ':' + ','.join(target)

with open('teacher.txt', 'w') as f:
    for i in range(4000):
        line = new_line()
        f.write(line)
        f.write('\n')
        print(line)

