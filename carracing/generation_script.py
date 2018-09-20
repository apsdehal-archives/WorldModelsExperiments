
"""
Encapsulate generate data to make it parallel
"""
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--threads', type=int, help="Number of threads")
args = parser.parse_args()

def _threaded_generation(i):
    start_episode = i * 1000
    cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    cmd += ['--server-num={}'.format(i + 1)]
    cmd += ["python", "model.py", "norender", "log/carracing.cma.16.64.best.json", str(start_episode)]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True

with Pool(args.threads) as p:
    p.map(_threaded_generation, range(args.threads))
