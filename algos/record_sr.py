import os

def record_test_sr(filepath: str = 'sr_log.txt', epoch: int = 0, sr: float = 0.):
    with open(filepath, 'a') as f:
        f.write(str(epoch) + ' ' + str(sr))
        f.write('\n')