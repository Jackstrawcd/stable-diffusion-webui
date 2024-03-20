import signal
import sys
import time
import os
from tools.environment import get_pod_status_env, set_pod_status_env
import logging
from tools.redis import RedisPool
from worker.executor import TaskExecutor, TaskProgress
from worker.task import TaskStatus

TERMINATING_STATUS = "terminating"
RUNNING_STATUS = "running"
SHORT_TASK = [1, 2, 3, 5, 6, 7, 8, 9, 10]
LONG_TASK = [4, 11]
WAIT_TASK_REDIS_KEY = "WAIT-TASK"


def signal_handler(task_exec: TaskExecutor):
    logging.info("接收退出信号,检查任务执行情况")
    # 设置环境变量，停止拿取任务
    set_pod_status_env(TERMINATING_STATUS)
    # 处理退出
    current_task = task_exec.current_task
    if current_task is None:
        logging.info("当前无任务执行，直接退出···")
        return
    task_type = current_task.task.task_type
    task_progress = current_task.task_progress
    redis_pool = RedisPool()
    redis = redis_pool.get_connection()
    if int(task_type) in SHORT_TASK:
        wait_task(current_task, redis)
    elif int(task_type) in LONG_TASK:
        if task_progress > 70:
            wait_task(current_task, redis)
        else:
            push_task(current_task, redis)
    else:
        logging.info("未知任务类型，等待执行完成")
        wait_task(current_task, redis)


def wait_task(current_task: TaskProgress, redis):
    """
    添加任务到等待队列，等待任务执行完成
    :param current_task:
    :param redis: redis
    :param task_id: 任务id
    :return:
    """
    # 任务ID添加到等待队列
    task_id = current_task.task.id
    redis.zadd(WAIT_TASK_REDIS_KEY, {task_id: current_task.task.create_at})
    while 1:
        task_progress = current_task.task_progress
        task_status = current_task.status
        if task_status not in [TaskStatus.Finish, TaskStatus.Failed]:
            logging.info(f"等待任务执行完成:task_id={task_id},task_progress={task_progress}")
            time.sleep(5)
        else:
            return


def push_task(current_task: TaskProgress, redis):
    """
    添加任务到重放队列
    :param redis:
    :param current_task:
    :param task_id: 任务id
    :return:
    """
    # 任务ID添加到等待队列
    task_id = current_task.task.id
    queue_name = current_task.task.get("queue")

    redis.zadd(queue_name, {current_task.task.json(): current_task.task.create_at})
    for i in range(5):
        logging.info(f"repush task {task_id}, Exit in {i} seconds ")
        time.sleep(1)
    return

