import time
import pynvml
import psutil


class StepCounter(object):
    def __init__(self, handle=None, current_process=None):
        self.time = 0
        self.cost_time = 0
        self.cost_memory = 0
        self.handle = handle
        self.cpu_usage = 0
        self.process = current_process

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cost_time = round(time.perf_counter() - self.time, 3)
        if self.process is not None:
            self.cpu_usage = self.process.cpu_percent(interval=1)
        else:
            self.cpu_usage = psutil.cpu_percent(interval=1)
        if self.handle is not None:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.cost_memory = memory_info.used / 1024 ** 2
        self.cost_time = cost_time


class TrainMonitor(object):
    def __init__(self):
        self.refs = []
        self.trans = []
        self.memory = []
        self.trans_with_info = []

        self.total_cost_time = 0
        self.total_audio_time = 0
        self.max_cpu_usage = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        rtf = round(self.total_cost_time / self.total_audio_time, 3)
        memory_max = max(self.memory)
        memory_avg = sum(self.memory) / len(self.memory)

        self.trans_with_info.append(f'total cost time: {self.total_cost_time}s')
        self.trans_with_info.append(f'RTF:             {rtf}')
        self.trans_with_info.append(f'Throughput:      {round(1 / rtf, 3)}')
        self.trans_with_info.append(f'Avg memory:      {memory_avg}')
        self.trans_with_info.append(f'Max memory:      {memory_max}')
        self.trans_with_info.append(f'Max cpu usage:   {self.max_cpu_usage}')


if __name__ == '__main__':
    pass

