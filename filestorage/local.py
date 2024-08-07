#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/20 4:08 PM
# @Author  : wangdongming
# @Site    : 
# @File    : local.py
# @Software: xingzhe.ai
import os
import shutil
from tools.environment import get_share_dir
from filestorage.storage import FileStorage


class LocalFileStorage(FileStorage):

    def __init__(self):
        super(LocalFileStorage, self).__init__()
        self.share_dir = get_share_dir()
        self.enable = self.share_dir and os.path.isdir(self.share_dir)

    def name(self):
        return "local"

    def download(self, remoting_path, local_path, progress_callback=None) -> str:
        if not self.enable:
            raise OSError('local fs disabled')
        dirname = os.path.dirname(remoting_path)
        if dirname != self.share_dir:
            remoting_path = os.path.join(self.share_dir, os.path.basename(remoting_path))
        shutil.copy(remoting_path, local_path)
        return local_path

    def upload(self, local_path, remoting_path) -> str:
        dirname = os.path.dirname(remoting_path)
        if dirname != self.share_dir:
            remoting_path = os.path.join(self.share_dir, os.path.basename(remoting_path))
        shutil.copy(local_path, remoting_path)
        return remoting_path

    def lock_download(self, remoting_path, local_path, progress_callback=None, expire=1800, flocker=True) -> str:
        return self.download(remoting_path, local_path, progress_callback)