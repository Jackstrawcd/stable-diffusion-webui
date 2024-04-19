#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/19 2:43 PM
# @Author  : wangdongming
# @Site    : 
# @File    : multi_digital.py
# @Software: xingzhe.ai
import random
import shutil
import typing
import os
from enum import IntEnum
from handlers.txt2img import Txt2ImgTask
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, ADetailer, mk_tmp_dir
from handlers.utils import get_tmp_local_path
from loguru import logger


class DigitalTaskType(IntEnum):
    Img2Img = 2  # 原本是1
    Txt2Img = 1  # 原本是2


class MultiGenDigitalPhotoLoraMeta:

    def __init__(self, image_keys: typing.Sequence[str], lora_key: str):
        image_keys = image_keys or []
        self.lora_key = lora_key
        self.images = []
        self.dirname = mk_tmp_dir(lora_key[:8])
        for key in image_keys:
            tmp = get_tmp_local_path(key)
            if not tmp:
                continue
            # 移动到特定目录下。
            local = os.path.join(self.dirname, os.path.basename(tmp))
            shutil.move(tmp, local)
            self.images.append(local)


class MultiGenDigitalPhotoTask(Txt2ImgTask):

    def __init__(self, base_model_path: str,
                 user_id: str,
                 lora_meta_array: typing.Sequence[typing.Mapping],  # LORA1, LORA2配置
                 default_script_arg_txt2img: typing.Sequence,  # 默认脚本参数，handler构造。
                 prompt: str,  # TAG
                 negative_prompt: str,  # 反向TAG
                 sampler_name: str = None,  # 采样器
                 enable_hr: bool = False,
                 denoising_strength: float = 0.7,  # 重绘幅度
                 tiling: bool = False,  # 可平铺
                 hr_scale: float = 2.0,
                 hr_upscaler: str = None,
                 hr_second_pass_steps: int = 0,
                 hr_resize_x: int = 0,
                 hr_resize_y: int = 0,
                 cfg_scale: float = 7.0,  # 提示词相关性
                 width: int = 512,  # 图宽
                 height: int = 512,  # 图高
                 restore_faces: bool = False,  # 面部修复
                 seed: int = -1,  # 随机种子
                 seed_enable_extras: bool = False,  # 是否启用随机种子扩展
                 subseed: int = -1,  # 差异随机种子
                 subseed_strength: float = 0,  # 差异强度
                 seed_resize_from_h: int = 0,  # 重置尺寸种子-高度
                 seed_resize_from_w: int = 0,  # 重置尺寸种子-宽度
                 batch_size: int = 1,  # 批次数量
                 n_iter: int = 1,  # 每个批次数量
                 steps: int = 30,  # 步数
                 select_script_name: str = None,  # 选择下拉框脚本名
                 select_script_args: typing.Sequence = None,  # 选择下拉框脚本参数
                 select_script_nets: typing.Sequence[typing.Mapping] = None,  # 选择下拉框脚本涉及的模型信息
                 alwayson_scripts=None,  # 插件脚本，object格式： {插件名: {'args': [参数列表]}}
                 override_settings_texts=None,  # 自定义设置 TEXT,如: ['Clip skip: 2', 'ENSD: 31337', 'sd_vae': 'None']
                 lora_models: typing.Sequence[str] = None,  # 使用LORA，用户和系统全部LORA列表
                 embeddings: typing.Sequence[str] = None,  # embeddings，用户和系统全部mbending列表
                 lycoris_models: typing.Sequence[str] = None,  # lycoris，用户和系统全部lycoris列表
                 compress_pnginfo: bool = True,  # 使用GZIP压缩图片信息（默认开启）
                 hr_sampler_name: str = None,  # hr sampler
                 hr_prompt: str = None,  # hr prompt
                 hr_negative_prompt: str = None,  # hr negative prompt
                 disable_ad_face: bool = True,  # 关闭默认的ADetailer face
                 enable_refiner: bool = False,  # 是否启用XLRefiner
                 refiner_switch_at: float = 0.2,  # XL 精描切换时机
                 refiner_checkpoint: str = None,  # XL refiner模型文件
                 **kwargs):

        super(MultiGenDigitalPhotoTask, self).__init__(
            base_model_path,
            user_id,
            default_script_arg_txt2img,
            prompt,
            negative_prompt,
            sampler_name,
            enable_hr,
            denoising_strength,
            tiling,
            hr_scale,
            hr_upscaler,
            hr_second_pass_steps,
            hr_resize_x,
            hr_resize_y,
            cfg_scale,
            width,
            height,
            restore_faces,
            seed,
            seed_enable_extras,
            subseed,
            subseed_strength,
            seed_resize_from_h,
            seed_resize_from_w,
            batch_size,
            n_iter,
            steps,
            select_script_name,
            select_script_args,
            select_script_nets,
            alwayson_scripts,
            override_settings_texts,
            lora_models,
            embeddings,
            lycoris_models,
            compress_pnginfo,
            hr_sampler_name,
            hr_prompt,
            hr_negative_prompt,
            disable_ad_face,
            enable_refiner,
            refiner_switch_at,
            refiner_checkpoint,
            **kwargs)
        self.lora_meta_map = {}
        if len(lora_meta_array) != 2:
            raise ValueError('lora count error')

        for i, meta in enumerate(lora_meta_array):
            lora_key = meta['lora_key']
            images = meta['image_keys']
            self.lora_meta_map[i] = MultiGenDigitalPhotoLoraMeta(images, lora_key)

    def get_train_image_local_dir(self):
        '''
        获取训练LORA的数据集本地目录，返回一个长度为2的数组，其索引0，1对应值分别代表左右LORA的训练集本地目录。
        如果下载失败（网络问题，数据集数据丢失或数据集过期等），对应元素对None。
        '''
        train_dataset = [None, None]
        for i in range(len(self.lora_meta_map)):
            images = self.lora_meta_map[i].images
            if images:
                train_dataset[i] = self.lora_meta_map[i].dirname

        return train_dataset

    @classmethod
    def from_task(cls, task: Task, default_script_args: typing.Sequence, refiner_checkpoint: str = None):
        base_model_path = task['base_model_path']
        alwayson_scripts = task['alwayson_scripts']
        user_id = task['user_id']
        select_script = task.get('select_script')
        select_script_name, select_script_args = None, None
        prompt = task.get('prompt', '')
        negative_prompt = task.get('negative_prompt', '')
        lora_meta_array = task.pop('lora_meta_array')

        if select_script:
            if not isinstance(select_script, dict):
                raise TypeError('select_script type err')
            select_script_name = select_script['name']
            select_script_args = select_script['args']
        else:
            select_script_name = task.get('select_script_name')
            select_script_args = task.get('select_script_args')

        kwargs = task.data.copy()
        kwargs.pop('base_model_path')
        kwargs.pop('alwayson_scripts')
        kwargs.pop('prompt')
        kwargs.pop('negative_prompt')
        kwargs.pop('user_id')
        if 'select_script' in kwargs:
            kwargs.pop('select_script')
        if 'select_script_name' in kwargs:
            kwargs.pop('select_script_name')
        if 'select_script_args' in kwargs:
            kwargs.pop('select_script_args')
        if 'lora_meta_array' in kwargs:
            kwargs.pop('lora_meta_array')

        if "nsfw" in prompt.lower():
            prompt = prompt.lower().replace('nsfw', '')
        kwargs['refiner_checkpoint'] = refiner_checkpoint

        return cls(base_model_path,
                   user_id,
                   default_script_args,
                   prompt=prompt,
                   negative_prompt=negative_prompt,
                   alwayson_scripts=alwayson_scripts,
                   select_script_name=select_script_name,
                   select_script_args=select_script_args,
                   lora_meta_array=lora_meta_array,
                   **kwargs)

