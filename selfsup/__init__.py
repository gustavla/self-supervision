import time

_start_time = time.time()

def timesup(limit):
    global _start_time
    if limit is None:
        return False
    else:
        return time.time() - _start_time > 3600 * limit
