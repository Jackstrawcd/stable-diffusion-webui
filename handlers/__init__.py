#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 12:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : __init__.py.py
# @Software: Hifive
import os
from worker.handler import TaskHandler, DumpTaskHandler
from tools.reflection import find_classes
from handlers.img2img import Img2ImgTaskHandler
from handlers.txt2img import Txt2ImgTaskHandler
from modelscope import snapshot_download
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def patch_tokenizer_base():
    """ Monkey patch PreTrainedTokenizerBase.from_pretrained to adapt to modelscope hub.
    """
    ori_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.bin', r'\w+\.safetensors']
        if "use_modelscope" in kwargs:
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', None)
                model_dir = snapshot_download(
                    pretrained_model_name_or_path,
                    revision=revision,
                    ignore_file_pattern=ignore_file_pattern)
            else:
                model_dir = pretrained_model_name_or_path
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)
        else:
            model_dir = pretrained_model_name_or_path
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

    PreTrainedTokenizerBase.from_pretrained = from_pretrained


def patch_model_base():
    """ Monkey patch PreTrainedModel.from_pretrained to adapt to modelscope hub.
    """
    ori_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.safetensors']
        if "use_modelscope" in kwargs:
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', None)
                model_dir = snapshot_download(
                    pretrained_model_name_or_path,
                    revision=revision,
                    ignore_file_pattern=ignore_file_pattern)
            else:
                model_dir = pretrained_model_name_or_path
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)
        else:
            model_dir = pretrained_model_name_or_path
            return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)

    PreTrainedModel.from_pretrained = from_pretrained


def get_task_handlers():
    for cls in find_classes("handlers"):
        if issubclass(cls, TaskHandler) and cls != TaskHandler and cls != DumpTaskHandler:
            yield cls()


def_task_handlers = [
    Img2ImgTaskHandler(),
    Txt2ImgTaskHandler()
]
print("hook patch_tokenizer_base & patch_model_base")
patch_tokenizer_base()
patch_model_base()
