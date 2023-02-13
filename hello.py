"""
A simple example
"""

import numpy as np
import datetime
import time
from datetime import timezone
import tensorflow as tf

secs = time.time()
dt_today = datetime.datetime.today()
print(f'Today is {dt_today}')
dt_now = datetime.datetime.now()
print(f'Today is {dt_now}')

a = np.array([1,2])
b = np.array([3,4])

print(f'a+b={a+b}')

#
# Simple tensorflow program to compare cpu and gpu execution times.
#

def matrix_multiply(size: int)-> dict:
    """
    Time a 2D matrix multiply using tensorflow accross all physical devices.
    
    Args:
     - size (int): The matrix size.
     - returns (dict): {Device string: elapsed time}
     """
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    results = {}
    for dev in tf.config.list_physical_devices():
        with tf.device(dev.device_type):
                ta = tf.constant(a)
                tb = tf.constant(b)
                t0 = time.time()
                x = tf.matmul(ta, tb)
                t1 = time.time()
                results.update({dev.device_type:t1 - t0})

   
    return results

print(f'{tf.config.list_physical_devices()}')
print(f'Matrix Multiply Elapsed Time: {matrix_multiply(4096)}')