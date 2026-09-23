import os
import subprocess
import datetime

import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"


def generate_run_commands(command_list=None,
                          array_command=None, array_indices=None, array_throttle=None,
                          n_cpus=1, n_gpus=0, dry=False, only_estimate_time=False,
                          mem=3000, length=None, hours=None, mins=59, tmp=None,
                          mode='local', gpu_model=None, prompt=True, gpu_mtotal=None,
                          relaunch=False, relaunch_after=None, local_env=None, delay=None,
                          output_filename_prefix="slurm-", output_filename="", output_path_prefix=""):

    if only_estimate_time:
        dry = False

    # check if single or array
    is_array_job = array_command is not None and array_indices is not None
    assert (command_list is not None) or is_array_job
    if is_array_job:
        assert all([(ind.isdigit() if type(ind) != int else True) for ind in array_indices]), f"array indices must be positive ints but got `{array_indices}`"

    # add environment variables needed based on compute configurations
    env_str = ""
    if n_gpus == 0:
        env_str += "JAX_PLATFORMS=cpu "

    # for avoiding numba race conditions
    env_str += "NUMBA_CACHE_DIR=$TMPDIR "
    env_str += "NUMBA_DISABLE_CACHE=1 "  # avoids crashes of GEARS

    if command_list is not None:
        command_list = [f"{env_str} {cmd}" for cmd in command_list]
    if array_command is not None:
        array_command = f"{env_str} {array_command}"

    if mode == "local":
        if prompt and not dry:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if is_array_job:
            command_list = [array_command.replace("\$SLURM_ARRAY_TASK_ID", str(ind)) for ind in array_indices]

        if answer == 'yes':
            print()
            for cmd in command_list:
                if dry:
                    print(cmd, end="\n")
                else:
                    osenv = os.environ.copy()
                    if local_env is not None:
                        osenv.update(local_env)
                    subprocess.call(cmd, shell=True, env=osenv)

    else:
        raise NotImplementedError

