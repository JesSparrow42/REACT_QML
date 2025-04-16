import torch.multiprocessing as mp

def pytest_configure(config):
    mp.set_start_method("spawn", force=True)
