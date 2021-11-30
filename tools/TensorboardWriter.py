import threading
from tensorboardX import SummaryWriter


class SingleSummaryWriter(SummaryWriter):
    _instance_lock = threading.Lock()

    def __init__(self, logdir=None, **kwargs):
        super().__init__(logdir, **kwargs)

    def __new__(cls, *args, **kwargs):
        if not hasattr(SingleSummaryWriter, "_instance"):
            with SingleSummaryWriter._instance_lock:
                if not hasattr(SingleSummaryWriter, "_instance"):
                    SingleSummaryWriter._instance = object.__new__(cls)
        return SingleSummaryWriter._instance
