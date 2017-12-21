# A thread [th] start with [th.start()]
# [start()] invokes [run()]
# Register [method] with [Thread(method)] or override [run()]
# [is_alive()] is True when [run()] starts, False when end
# [th2.join(th1)] blocks calling the thread
# [th1.setName("hoge")] changes the name

import time
import random
from threading import Thread
def printh(name):
  for i in range(3):
    time.sleep(random.random() * 10)
    print(name, i)

th2 = Thread(target=printh, args=('John',))
th1 = Thread(target=printh, args=('Kate',))
th1.start()
th2.start()
while True:
  time.sleep(1)
  th1_a = th1.isAlive()
  th2_a = th2.isAlive()
  print(th1_a, th2_a)
  if (not th1_a) and (not th2_a) and input() == 'again':
    th2 = Thread(target=printh, args=('John',))
    th1 = Thread(target=printh, args=('Kate',))
    th1.start()
    th2.start()
