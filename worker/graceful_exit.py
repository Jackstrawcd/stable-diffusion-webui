import signal
import sys
import time
import os
from tools.environment import get_pod_status_env, set_pod_status_env,get_wait_task_env,set_wait_task_env
from loguru import logger
from tools.redis import RedisPool
from worker.executor import TaskProgress
from worker.task import TaskStatus
import json
from threading import Event

TERMINATING_STATUS = "terminating"
RUNNING_STATUS = "running"
SHORT_TASK = [1, 2, 3, 5, 6, 7, 8, 9, 10]
LONG_TASK = [4, 11]
WAIT_TASK_REDIS_KEY = "WAIT-TASK"

event = Event()
def set_event():
    try:
        logger.info(f"trigger complete at:{time.ctime()}" )
        event.set()
    except Exception as err:
        print(err)

def wait_event():
    event.wait()  # Blocks until the flag becomes true.
    logger.info(f"Wait complete at:{time.ctime()}" )
    event.clear()  # Resets the flag.


def is_wait_task(current_task: TaskProgress):
    # 处理退出
    if current_task is None or current_task.status  in [TaskStatus.Finish, TaskStatus.Failed]:
        logger.info("当前无任务执行，直接退出···")
        return 
    task_type = current_task.task.task_type
    task_progress = current_task.task_progress
    if int(task_type) in LONG_TASK and task_progress < 70:
        push_task(current_task)
    else:
        wait_task(current_task)


def wait_task(current_task: TaskProgress):
    """
    添加任务到等待队列，等待任务执行完成
    :param current_task:
    :param redis: redis
    :param task_id: 任务id
    :return:
    """
    # 任务ID添加到等待队列
    if not  get_wait_task_env():
        redis_pool = RedisPool()
        redis = redis_pool.get_connection() 
        redis.zadd(WAIT_TASK_REDIS_KEY, {current_task.task.id: current_task.task.create_at})
        logger.info(f"task push wait queue,task_id={current_task.task.id},process={current_task.task_progress}")
        redis.close()
        set_wait_task_env(task_id=current_task.task.id)
    logger.info(f"wait task finish,task_id={current_task.task.id},process={current_task.task_progress}")


def push_task(current_task: TaskProgress, redis):
    """
    添加任务到重放队列
    :param redis:
    :param current_task:
    :param task_id: 任务id
    :return:
    """
    # 任务ID添加到等待队列
    redis.zadd(current_task.task.get("queue"), {current_task.task.id: current_task.task.create_at})
    old_task=current_task.task.json()
    redis.set(current_task.task.id,old_task)
    redis.close()
    set_event()
    logger.info(f"repush task {current_task.task.id}")
    while 1 :
        logger.info("The thread is blocked, waiting for exit")
        time.sleep(10)

