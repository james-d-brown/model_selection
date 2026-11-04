"""Thread-parallel execution of independent numerical tasks.

The nonlinear tier's cost is dominated by eigendecompositions of n x n kernel
matrices: one per trial bandwidth of each conditioning regression, one per
variable's feature kernel, and those of each candidate's effect size. They are
independent, and LAPACK releases Python's global interpreter lock, so they
run side by side in threads, each with a single linear-algebra thread
(threadpoolctl). Single threads are required: OpenBLAS keeps one shared pool
of worker threads, and several Python threads using it with more than one
thread each can crash it. They are also faster: one eigendecomposition
spreads poorly over many cores.

Regions nest at most once. A region opened inside a task of an outer region
(the eigendecompositions inside one comparison, say) sends its tasks to a
shared pool of inner workers, so outer tasks running side by side still
spread their inner work over every core. Inner tasks never wait on other
tasks, so the two pools cannot deadlock; a region opened inside an inner
task runs in sequence. The linear-algebra limit is set once, by the
outermost region, and restored when it ends.
Results do not depend on the number of workers, apart from rounding: the
linear-algebra library sums in a different order with a different number of
threads.

set_workers(n) sets the number of workers (None: every core; 1: no
parallelism); SelectionConfig.n_jobs sets it for a run.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

_WORKERS: int | None = None
_LOCAL = threading.local()


def set_workers(n: int | None) -> None:
    """Workers for parallel_map: None for every core, 1 to run in sequence."""
    global _WORKERS
    if n is not None and n < 1:
        raise ValueError(f"the number of workers must be at least 1 -- got {n}")
    _WORKERS = n


def workers() -> int:
    return _WORKERS or os.cpu_count() or 1


def _wrap(func: Callable, depth: int, inner: ThreadPoolExecutor | None) -> Callable:
    def task(item):
        _LOCAL.depth, _LOCAL.inner = depth, inner
        try:
            return func(item)
        finally:
            _LOCAL.depth, _LOCAL.inner = 0, None
    return task


def parallel_map(func: Callable, items: Iterable) -> list:
    """[func(item) for item in items], in order, with independent items run
    in parallel threads when more than one worker is available."""
    items = list(items)
    n = min(workers(), len(items))
    depth = getattr(_LOCAL, "depth", 0)
    if n <= 1 or depth >= 2:
        return [func(item) for item in items]
    if depth == 1:
        inner = getattr(_LOCAL, "inner", None)
        if inner is None:
            return [func(item) for item in items]
        return list(inner.map(_wrap(func, 2, None), items))

    try:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # still correct, but tasks may share linear-algebra threads
        limits = None
    try:
        with ThreadPoolExecutor(max_workers=workers()) as inner, \
                ThreadPoolExecutor(max_workers=n) as outer:
            return list(outer.map(_wrap(func, 1, inner), items))
    finally:
        if limits is not None:
            limits.restore_original_limits()


def run_tasks(tasks: list) -> list:
    """Run independent zero-argument tasks in one pool; results in order."""
    return parallel_map(lambda f: f(), tasks)


def batches(items: list, tasks_per_item: int) -> list[list]:
    """items in consecutive groups whose tasks fill the workers about once,
    so that memory in flight grows with the number of cores, not items."""
    size = max(1, -(-workers() // max(tasks_per_item, 1)))
    return [items[i:i + size] for i in range(0, len(items), size)]
