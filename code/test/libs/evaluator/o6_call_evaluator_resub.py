import numpy as np
import torch
import torch.multiprocessing as mp
import time


def evaluate_process(
    return_queue
):
    from o5_evaluator_resub import online_inference
    arr = online_inference()
    return_queue.put((arr))


def call_func():
    st = time.time()
    torch.multiprocessing.set_start_method('spawn', force=True)
    return_queue = mp.SimpleQueue()

    p = mp.Process(
        target=evaluate_process,
        args=(return_queue,)
    )
    p.start()

    raw_results = [return_queue.get()]  # list of tuple

    p.join()
    arr = raw_results[0]
    infer_time = time.time() - st
    print(f"time: {infer_time}")
    # print(arr)
    return arr


if __name__ == '__main__':
    call_func()
