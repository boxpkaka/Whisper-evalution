import time
import pynvml


class CountTime(object):
    def __init__(self, handle=None):
        self.time = 0
        self.cost_time = 0
        self.cost_memory = 0
        self.handle = handle

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cost_time = round(time.perf_counter() - self.time, 3)
        if self.handle is not None:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.cost_memory = memory_info.used / 1024 ** 2
        self.cost_time = cost_time


class TrainMonitor:
    def __init__(self, handle=None):
        self.time = 0
        self.cost_time = 0
        self.cost_memory = 0
        self.handle = handle

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cost_time = round(time.perf_counter() - self.time, 3)
        if self.handle is not None:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.cost_memory = memory_info.used / 1024 ** 2
        self.cost_time = cost_time

if __name__ == '__main__':
    pass

