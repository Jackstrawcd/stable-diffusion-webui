import signal
import sys
import time
import os
from tools.environment import get_pod_status_env, set_pod_status_env
import logging
from tools.redis import RedisPool
from worker.executor import TaskExecutor

TERMINATING_STATUS = "terminating"
RUNNING_STATUS = "running"
SHORT_TASK = [1, 2, 3, 5, 6, 7, 8, 9, 10]
LONG_TASK = [4, 11]
WAIT_TASK_REDIS_KEY = "WAIT-TASK"


def signal_handler(task: TaskExecutor):
    logging.info("接收退出信号,检查任务执行情况")
    # 设置环境变量，停止拿取任务
    set_pod_status_env(TERMINATING_STATUS)
    # 处理退出
    task_id = "task."
    task_type = ""
    task_progress = ""
    redis_pool = RedisPool()
    redis = redis_pool.get_connection()
    if task_id is None:
        logging.info("当前无任务执行，直接退出···")
        return
    if int(task_type) in SHORT_TASK:
        wait_task(task_id, redis)
    elif int(task_type) in LONG_TASK:
        if task_progress > 70:
            wait_task(task_id, redis)
        else:
            push_task(task_id, redis)
    else:
        logging.info("未知任务类型，等待执行完成")
        wait_task(task_id, redis)


def wait_task(task_id, redis):
    """
    添加任务到等待队列，等待任务执行完成
    :param redis: redis
    :param task_id: 任务id
    :return:
    """
    # 任务ID添加到等待队列
    redis.zadd(WAIT_TASK_REDIS_KEY, {task_id: int(time.time())})
    while 1:
        task_progress = get_task_progress_env()
        if task_progress < 100:
            logging.info(f"等待任务执行完成:task_id={task_id},task_progress={task_progress}")
            time.sleep(5)
        else:
            return


def push_task(task_id, redis):
    """
    添加任务到重放队列
    :param task_id: 任务id
    :return:
    """
    pass


if __name__ == '__main__':
    # 注册信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    print('按下 Ctrl+C 来测试信号处理程序...')
