import time
import pynvml


class CountTime(object):
    def __init__(self, handle):
        self.time = 0
        self.cost_time = 0
        self.cost_memory = 0
        self.handle = handle

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cost_time = round(time.perf_counter() - self.time, 3)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.cost_time = cost_time
        self.cost_memory = memory_info.used / 1024**2


if __name__ == '__main__':
    x = 0

    time1 = time.perf_counter()
    for i in range(900):
        with CountTime() as ct:
            x += 1
    time2 = time.perf_counter()

    for i in range(900):
        with CountTime() as ct:
            x += 1
    time3 = time.perf_counter()

    print(time3 - time2 - time2 + time1)

