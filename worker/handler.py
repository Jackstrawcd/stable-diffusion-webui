#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 9:54 AM
# @Author  : wangdongming
# @Site    : 
# @File    : handler.py
# @Software: Hifive
import abc
import time
import typing
import traceback
import torch.cuda
from modules.shared import mem_mon as vram_mon
from worker.dumper import dumper
from loguru import logger
from modules.devices import torch_gc, get_cuda_device_string
from worker.k8s_health import write_healthy, system_exit
from worker.task import Task, TaskProgress, TaskStatus, TaskType
try:
    from collections.abc import Iterable
    from collections import defaultdict
except ImportError:
    from collections import defaultdict, Iterable

RegisterMinorHandlerArg = typing.Tuple[int, typing.Callable]
RegisterMinorHandlerArgWithArgs = typing.Tuple[int, typing.Callable, typing.Sequence]
RegisterMinorHandlerArgWithKwargs = typing.Tuple[int, typing.Callable, typing.Mapping]
RegisterMinorHandlerArgWithArgsKwargs = typing.Tuple[int, typing.Callable, typing.Sequence, typing.Mapping]

RegisterMinorHandlerArgType = typing.Union[
    RegisterMinorHandlerArg,
    RegisterMinorHandlerArgWithArgs,
    RegisterMinorHandlerArgWithKwargs,
    RegisterMinorHandlerArgWithArgsKwargs
]


class TaskHandler:

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.enable = True
        self.minor_handlers = defaultdict(dict)

    def handle_task_type(self):
        return self.task_type

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        meta = self._get_minor_handler(task.minor_type)
        if not meta:
            raise OSError(f'cannot found task minor type handler, type:{task.task_type}, minor:{task.minor_type}')

        args = meta.get('args') or []
        kwargs = meta.get('kwargs') or {}
        func = meta['func']
        args.insert(0, task)

        yield from func(*args, **kwargs)

    def _register_minor_handler_meta(self, v: RegisterMinorHandlerArgType):
        f = v[1]
        minor_type = int(v[0])
        d = {
            'func': f,
            "minor_type": minor_type,
        }

        if len(v) > 2:
            for i in range(1, 3):
                x = v[-i]
                if isinstance(x, dict):
                    d['kwargs'] = x
                elif isinstance(x, Iterable):
                    d['args'] = x
        self.minor_handlers[minor_type] = d

    def _get_minor_handler(self, minor_type: int):
        minor_type = minor_type if minor_type > 0 else 1
        return self.minor_handlers.get(int(minor_type))

    def _set_task_status(self, p: TaskProgress):
        logger.info(f">>> task:{p.task.desc()}, status:{p.status.name}, desc:{p.task_desc}")

    def do(self, task: Task, progress_callback=None):
        ok, msg = task.valid()
        if not ok:
            p = TaskProgress.new_failed(task, msg)
            self._set_task_status(p)
        else:
            try:
                if not self.can_do(task):
                    raise OSError(f'handler({task.task_type}) unsupported minor type:{task.minor_type}')

                p = TaskProgress.new_prepare(task, msg)
                self._set_task_status(p)
                for progress in self._exec(task):
                    # 判断是不是重新执行，不再更新比上一次进度小的
                    old_task_progress = progress.task.get('old_task_progress', 0)
                    if progress.task_progress < old_task_progress:
                        progress.task_progress = old_task_progress

                    self._set_task_status(progress)
                    if callable(progress_callback):
                        progress_callback(progress)

            except torch.cuda.OutOfMemoryError:
                ok = torch_gc()
                time.sleep(15)
                logger.exception('CUDA out of memory')
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
                p = TaskProgress.new_failed(
                    task,
                    'CUDA out of memory',
                    f'CUDA out of memory and release, free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                self._set_task_status(p)
                progress_callback(p)
                system_exit(free, total, coercive=not ok)
            except (RuntimeError, torch.cuda.CudaError) as runtimeErr:
                trace = traceback.format_exc()
                msg = str(runtimeErr)
                logger.exception('unhandle runtime err')
                p = TaskProgress.new_failed(task, msg, trace)
                self._set_task_status(p)
                progress_callback(p)
                free, total = vram_mon.cuda_mem_get_info()
                time.sleep(15)
                system_exit(free, total, coercive=True)
            except Exception as ex:
                trace = traceback.format_exc()
                msg = str(ex)
                logger.exception('unhandle err')
                p = TaskProgress.new_failed(task, msg, trace)

                self._set_task_status(p)
                progress_callback(p)
                if not torch_gc():
                    free, total = vram_mon.cuda_mem_get_info()
                    system_exit(free, total, coercive=True)
                if 'BrokenPipeError' in str(ex):
                    pass

    def register(self, *args: RegisterMinorHandlerArgType):
        for item in args:
            f = item[1]
            if not callable(f):
                raise ValueError('f can not callable')
            self._register_minor_handler_meta(item)

    def can_do(self, task: Task):
        if task.task_type != self.task_type:
            return False
        h = self._get_minor_handler(task.minor_type)
        return isinstance(h, dict)

    def close(self):
        pass

    def set_failed(self, task: Task, desc: str):
        p = TaskProgress.new_failed(task, desc)
        self._set_task_status(p)

    def __call__(self, *args, **kwargs):
        return self.do(*args, **kwargs)


class DumpTaskHandler(TaskHandler, abc.ABC):

    def __init__(self, task_type: TaskType):
        super(DumpTaskHandler, self).__init__(task_type)

    def _set_task_status(self, p: TaskProgress):
        super()._set_task_status(p)
        dumper.dump_task_progress(p)

    def close(self):
        dumper.stop()
