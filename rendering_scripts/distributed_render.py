"""

Modified from https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/distributed.py

"""

import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os
import megfile
import tyro
import wandb
import time
import threading


file_lock = threading.Lock()

@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""

    tag: str = "default"
    """Tag to differentiate different versions"""

    resolution: int = 1024
    """Rendering resolution"""


    timeout: int = 3600

    render_space: str = "VIEW"

    objects_uid: str = f"./dataset/material3d.json"

    timeout_uid: str = "./rendering_scripts/timeout_uid.txt"
    
    rendered_uid: str = "./rendering_scripts/rendered_uid.txt"

    local_output_dir: str = "./dataset/outputs/"



def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    # s3: Optional[boto3.client],
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        uid = item.split("/")[-1].split(".")[0]

        view_path = os.path.join(args.local_output_dir, uid)
        os.makedirs(view_path, exist_ok = True)

        # Perform some operation on the item
        print(item, gpu)

        try:
            command = (
                # f"export DISPLAY=:0.{gpu} &&"
                # f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
                f" CUDA_VISIBLE_DEVICES={gpu} "
                f" blender -b -P rendering_scripts/blender_script_material.py --"
                f" --object_path {item}"
                f" --output_dir {args.local_output_dir}"
                f" --resolution {args.resolution}"
                f" --render_space {args.render_space}"
            )
            print(command)
            result = subprocess.run(command, shell=True, timeout=args.timeout)
            if result.returncode != 0:
                print(f"Blender process for item {item} returned non-zero exit code {result.returncode}")   
                if megfile.smart_exists(view_path):
                    megfile.smart_remove(view_path)       
            else:
                with file_lock:
                    with open(args.rendered_uid, "a") as f:
                        f.write(uid + "\n")
        except subprocess.TimeoutExpired:
            print(f"Blender process for item {item} timed out after {args.timeout} seconds")
            if megfile.smart_exists(view_path):
                megfile.smart_remove(view_path)
            with file_lock:
                with open(args.timeout_uid, "a") as f:
                    f.write(uid + "\n")
        finally:

            with count.get_lock():
                count.value += 1

            queue.task_done()
            time.sleep(0.02)


if __name__ == "__main__":

    start_i = time.time()

    args = tyro.cli(Args)


    # s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_list = list(map(int, visible_devices.split(","))) if visible_devices else list(range(args.num_gpus))

    print(gpu_list)
    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                # target=worker, args=(queue, count, gpu_i, s3)
                target=worker, args=(queue, count, gpu_list[gpu_i])
            )
            process.daemon = True
            process.start()
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)

    with open(args.objects_uid, "r") as f:
        objects_uid = json.load(f)

    with open(args.timeout_uid, "r") as f:
        timeout_uid = f.read().split('\n')[:-1]

    with open(args.rendered_uid, "r") as f:
        rendered_uid = f.read().split('\n')[:-1]

    objects_uid = set(objects_uid) - set(timeout_uid) - set(rendered_uid)
    
    if len(objects_uid) > 0:
        model_keys = list(objects_uid)
        model_keys = sorted(model_keys)
    
    print(f"Start from {model_keys[0]}")

    for item in model_keys:
        queue.put(model_paths[item])

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)

    
    # Wait for all processings to finish
    for p in multiprocessing.active_children():
        p.join()
    
    end_i = time.time()
    len_model_keys = len(model_keys)
    print(f"Finished all {len_model_keys} models in " + f"{end_i - start_i}" + " seconds !!!!!!!!!!!!!!!!!!!!!" )
