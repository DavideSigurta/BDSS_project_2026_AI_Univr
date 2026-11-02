"""Run one measuring function in a fresh process with a hard wall-clock limit."""

from __future__ import annotations

import multiprocessing
from queue import Empty
import threading


def run_limited(target, args, limit):
    """Run `target(*args, queue)` in a child; return its payload, None on timeout.

    A reader thread drains the queue while the parent joins: reading after the join would
    deadlock as soon as the child's payload fills the pipe.
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue), daemon=True)
    process.start()
    result = []
    stop = threading.Event()

    def read():
        while not stop.is_set():
            try:
                result.append(queue.get(timeout=0.05))
                return
            except Empty:
                continue
            except (EOFError, OSError):
                return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    process.join(limit)
    if process.is_alive():
        process.terminate()
        process.join()
        stop.set()
        reader.join(1.0)
        queue.close()
        try:
            queue.join_thread()
        except (OSError, ValueError):
            pass
        process.close()
        return None
    reader.join(1.0)
    stop.set()
    reader.join(1.0)
    queue.close()
    try:
        queue.join_thread()
    except (OSError, ValueError):
        pass
    exitcode = process.exitcode
    process.close()
    if exitcode not in (0, None):
        return ("error", f"child exited with {exitcode}")
    return result[0] if result else ("error", "child produced no result")
