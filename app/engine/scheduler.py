from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.engine.common import ImageJob, ImageResult, ModelNotLoadedError


CompleteFn = Callable[[ImageJob], ImageResult | Awaitable[ImageResult]]


class LoadedModelExecutor:
    def __init__(self, model_name: str, complete_fn: CompleteFn, target_inflight: int) -> None:
        self.model_name = model_name
        self._complete_fn = complete_fn
        self._target_inflight = max(1, target_inflight)
        self._queue: asyncio.Queue[tuple[ImageJob, float, asyncio.Future[ImageResult]]] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._inflight = 0
        self._closed = False

    @property
    def target_inflight(self) -> int:
        return self._target_inflight

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._workers:
            return
        self._closed = False
        for index in range(self._target_inflight):
            self._workers.append(asyncio.create_task(self._worker_loop(index)))

    async def stop(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        while not self._queue.empty():
            _job, _enqueued_at, future = self._queue.get_nowait()
            if not future.done():
                future.set_exception(ModelNotLoadedError(f"model unloaded: {self.model_name}"))

    async def submit(self, job: ImageJob) -> ImageResult:
        if self._closed:
            raise ModelNotLoadedError(f"model is not loaded: {self.model_name}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ImageResult] = loop.create_future()
        await self._queue.put((job, time.perf_counter(), future))
        return await future

    async def _worker_loop(self, _index: int) -> None:
        while True:
            job, enqueued_at, future = await self._queue.get()
            if future.cancelled():
                continue
            self._inflight += 1
            started_at = time.perf_counter()
            try:
                result = self._complete_fn(job)
                if inspect.isawaitable(result):
                    result = await result
                result.metrics.setdefault("engine_queue_wait_ms", (started_at - enqueued_at) * 1000)
                result.metrics.setdefault("engine_total_wall_ms", (time.perf_counter() - enqueued_at) * 1000)
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._inflight -= 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "target_inflight": self.target_inflight,
            "inflight": self.inflight,
            "queued": self.queued,
        }


class RuntimeScheduler:
    def __init__(self) -> None:
        self._executors: dict[str, LoadedModelExecutor] = {}
        self._lock = asyncio.Lock()

    async def register(self, model_name: str, executor: LoadedModelExecutor) -> None:
        await executor.start()
        async with self._lock:
            old_executor = self._executors.pop(model_name, None)
            self._executors[model_name] = executor
        if old_executor is not None:
            await old_executor.stop()

    async def unregister(self, model_name: str) -> None:
        async with self._lock:
            executor = self._executors.pop(model_name, None)
        if executor is not None:
            await executor.stop()

    async def complete(self, model_name: str, job: ImageJob) -> ImageResult:
        async with self._lock:
            executor = self._executors.get(model_name)
        if executor is None:
            raise ModelNotLoadedError(f"model is not loaded: {model_name}")
        return await executor.submit(job)

    async def close(self) -> None:
        async with self._lock:
            executors = list(self._executors.items())
            self._executors.clear()
        for _model_name, executor in executors:
            await executor.stop()

    def snapshot(self, model_name: str) -> dict[str, Any] | None:
        executor = self._executors.get(model_name)
        if executor is None:
            return None
        return executor.snapshot()
