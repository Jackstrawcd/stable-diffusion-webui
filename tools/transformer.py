#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 2:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : transformer.py
# @Software: xingzhe.ai
import os
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from modelscope import snapshot_download

logger.info(f"save PreTrainedTokenizerBase.from_pretrained: {PreTrainedTokenizerBase.from_pretrained}")

# 记录原本的from_pretrained语义
ori_tokenizer_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__
ori_model_from_pretrained = PreTrainedModel.from_pretrained.__func__

logger.info(f"save PreTrainedTokenizerBase.from_pretrained hash:{hash(ori_tokenizer_from_pretrained)}")


def patch_tokenizer_base():
    """ Monkey patch PreTrainedTokenizerBase.from_pretrained to adapt to modelscope hub.
    """
    ori_from_pretrained = ori_tokenizer_from_pretrained

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
    ori_from_pretrained = ori_model_from_pretrained

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

def hook():
    patch_tokenizer_base()
    patch_model_base()