"""
A simple example
"""

import numpy as np
import datetime
import time
from datetime import timezone

secs = time.time()
dt_today = datetime.datetime.today()
print(f'Today is {dt_today}')
dt_now = datetime.datetime.now()
print(f'Today is {dt_now}')

a = np.array([1,2])
b = np.array([3,4])

print(f'a+b={a+b}')
