#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 3:47 PM
# @Author  : wangdongming
# @Site    : 
# @File    : handler.py
# @Software: Hifive
import os
from enum import IntEnum
from worker.handler import DumpTaskHandler
from worker.task import Task, TaskType
from .preprocess import exec_preprocess_task
from .lora import exec_train_lora_task, start_train_process
from trainx.doppelganger import digital_doppelganger
from trainx.typex import TrainMinorTaskType
from modules.devices import torch_gc
from tools import safety_clean_tmp
from worker.dumper import MongoTaskDumper
from worker import graceful_exit
from tools.environment import get_pod_status_env

class TrainTaskHandler(DumpTaskHandler):

    def __init__(self):
        super(TrainTaskHandler, self).__init__(TaskType.Train)
        self.register(
            (TrainMinorTaskType.Preprocess, exec_preprocess_task),
            (TrainMinorTaskType.Lora, exec_train_lora_task, [self.dump_task]),
            (TrainMinorTaskType.DigitalDoppelganger, digital_doppelganger, [self.dump_task]),
        )

    def before_exec(self, task: Task, *args, **kwargs):
        torch_gc()
        safety_clean_tmp(3600)

    #
    # def _exec(self, task: Task):
    #     torch_gc()
    #     safety_clean_tmp(3600)
    #     print(f"current pid:{os.getpid()}")
    #     if task.minor_type == TrainMinorTaskType.Preprocess:
    #         yield from exec_preprocess_task(task)
    #     elif task.minor_type == TrainMinorTaskType.Lora:
    #         yield from exec_train_lora_task(task, self.dump_task)
    #     elif task.minor_type == TrainMinorTaskType.DigitalDoppelganger:
    #         yield from digital_doppelganger(task, self.dump_task)

    def dump_task(self, p):
        print(f"current pid:{os.getpid()}")
        if get_pod_status_env() == graceful_exit.TERMINATING_STATUS:
            graceful_exit.is_wait_task(p)
        with MongoTaskDumper() as dumper:
            dumper.dump_task_progress(p)
