import subprocess
import argparse


def get_gpu_info2():
    try:
        line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        line = line_as_bytes.decode("ascii")
        _, line = line.split(":", 1)
        # line, _ = line.split("(")
        return line.strip()
    except subprocess.CalledProcessError as e:
        return "No GPU info"


def str2bool(v):
    v = "".join([char for char in v if u"\xa0" not in char])  # avoid utf8 parsing error
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 'T', 't', 'Y', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'F', 'f', 'N', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

