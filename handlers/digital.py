#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 2:43 PM
# @Author  : wangdongming
# @Site    : 
# @File    : digital.py
# @Software: Hifive
import random
import re
import time
import typing
import uuid
import os
import shutil
import dlib
import cv2
import modules
import numpy as np
import modules.shared as shared
from enum import IntEnum
from PIL import ImageOps, Image
from handlers.txt2img import Txt2ImgTask
from handlers.multi_digital import MultiGenDigitalPhotoTask
from handlers.img2img import Img2ImgTask, Img2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, ADetailer, mk_tmp_dir
from handlers.utils import get_tmp_local_path, upload_files, upload_pil_image
from loguru import logger
from tools.wrapper import FuncExecTimeWrapper
from collections import defaultdict

from transformers import PreTrainedTokenizerBase, PreTrainedModel
from sd_scripts.library.transformers_pretrained import ori_tokenizer_from_pretrained, ori_model_from_pretrained

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download

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


patch_tokenizer_base()
patch_model_base()

def segment(segmentation_pipeline, img, ksize=0, eyeh=0, ksize1=0, include_neck=False, warp_mask=None,
            return_human=False):
    if True:
        result = segmentation_pipeline(img)
        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        if len(masks) == 0:
            return
        h, w = masks[0].shape
        mask_face = np.zeros((h, w))
        mask_hair = np.zeros((h, w))
        mask_neck = np.zeros((h, w))
        mask_cloth = np.zeros((h, w))
        mask_human = np.zeros((h, w))
        for i in range(len(labels)):
            if scores[i] > 0.8:
                if labels[i] == 'Torso-skin':
                    mask_neck += masks[i]
                elif labels[i] == 'Face':
                    mask_face += masks[i]
                elif labels[i] == 'Human':
                    mask_human += masks[i]
                elif labels[i] == 'Hair':
                    mask_hair += masks[i]
                elif labels[i] == 'UpperClothes' or labels[i] == 'Coat':
                    mask_cloth += masks[i]
        mask_face = np.clip(mask_face, 0, 1)
        mask_hair = np.clip(mask_hair, 0, 1)
        mask_neck = np.clip(mask_neck, 0, 1)
        mask_cloth = np.clip(mask_cloth, 0, 1)
        mask_human = np.clip(mask_human, 0, 1)
        if np.sum(mask_face) > 0:
            soft_mask = np.clip(mask_face, 0, 1)
            if ksize1 > 0:
                kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                kernel1 = np.ones((kernel_size1, kernel_size1))
                soft_mask = cv2.dilate(soft_mask, kernel1, iterations=1)
            if ksize > 0:
                kernel_size = int(np.sqrt(np.sum(soft_mask)) * ksize)
                kernel = np.ones((kernel_size, kernel_size))
                soft_mask_dilate = cv2.dilate(soft_mask, kernel, iterations=1)
                if warp_mask is not None:
                    soft_mask_dilate = soft_mask_dilate * (np.clip(soft_mask + warp_mask[:, :, 0], 0, 1))
                if eyeh > 0:
                    soft_mask = np.concatenate((soft_mask[:eyeh], soft_mask_dilate[eyeh:]), axis=0)
                else:
                    soft_mask = soft_mask_dilate
        else:
            if ksize1 > 0:
                kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                kernel1 = np.ones((kernel_size1, kernel_size1))
                soft_mask = cv2.dilate(mask_face, kernel1, iterations=1)
            else:
                soft_mask = mask_face
        if include_neck:
            soft_mask = np.clip(soft_mask + mask_neck, 0, 1)

    if return_human:
        mask_human = cv2.GaussianBlur(mask_human, (21, 21), 0) * mask_human
        return soft_mask, mask_human
    else:
        return soft_mask


def select_high_quality_face(input_img_dir):
    input_img_dir = str(input_img_dir)
    quality_score_list = []
    abs_img_path_list = []
    ## TODO
    face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa',
                                 model_revision='v2.0')

    for img_name in os.listdir(input_img_dir):
        if img_name.endswith('jsonl') or img_name.startswith('.ipynb') or img_name.startswith('.safetensors'):
            continue

        if img_name.endswith('jpg') or img_name.endswith('png'):
            abs_img_name = os.path.join(input_img_dir, img_name)
            face_quality_score = face_quality_func(abs_img_name)[OutputKeys.SCORES]
            if face_quality_score is None:
                quality_score_list.append(0)
            else:
                quality_score_list.append(face_quality_score[0])
            abs_img_path_list.append(abs_img_name)

    sort_idx = np.argsort(quality_score_list)[::-1]
    print('Selected face: ' + abs_img_path_list[sort_idx[0]])

    return Image.open(abs_img_path_list[sort_idx[0]])


def face_swap_fn(use_face_swap, gen_results, template_face):
    if use_face_swap:
        ## TODO
        out_img_list = []
        image_face_fusion = pipeline('face_fusion_torch',
                                     model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.5')
        segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
        for img in gen_results:
            result = image_face_fusion(dict(template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            face_mask = segment(segmentation_pipeline, img, ksize=0.1)
            result = (result * face_mask[:, :, None] + np.array(img)[:, :, ::-1] * (1 - face_mask[:, :, None])).astype(
                np.uint8)
            out_img_list.append(result)
        return out_img_list
    else:
        ret_results = []
        for img in gen_results:
            ret_results.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        return ret_results


class DigitalTaskType(IntEnum):
    Img2Img = 2  # 原本是1
    Txt2Img = 1  # 原本是2
    Multi = 3  # 多人


# class DigitalTaskHandler(Txt2ImgTaskHandler):
class DigitalTaskHandler(Img2ImgTaskHandler):

    def __init__(self):
        super(DigitalTaskHandler, self).__init__()
        self.task_type = TaskType.Digital
        self.register(
            (DigitalTaskType.Img2Img, self._exec_img2img),
            (DigitalTaskType.Txt2Img, self._exec_txt2img),
            (DigitalTaskType.Multi, self._exec_multi_portrait),
        )

    def _denoising_strengths(self, t: Task):

        def is_like_me():
            prompt = t['prompt']
            tokens = re.split("|".join(map(re.escape, ",")), prompt.replace(">", ">,").replace("<", ",<"))
            try:
                def parse_addition_networks(token: str):
                    if '<lora:' in token:
                        frags = token[1:-1].split(":")
                        entity = ':'.join(frags[:-1])
                        try:
                            weights = float(frags[-1])
                        except ValueError:
                            return None

                        # 查找对应LORA的instance id
                        lora_name = entity[len('lora:'):]
                        return lora_name, weights

                for token in tokens:
                    #  prompt相邻词之间有多个逗号
                    if not token.strip():
                        continue
                    r = parse_addition_networks(token.replace("\n", " ").strip())
                    if not r:
                        continue
                    return r[-1] > 0.8
            except:
                pass
            return False

        if is_like_me():
            return [0.45, 0.5, 0.5, 0.5]
        return [0.5, 0.5, 0.5, 0.5]

    def _get_init_images(self, t: Task):
        images = (t.get('init_img') or "").split(',')
        n_iter = t.get('n_iter') or 1
        batch_size = t.get('batch_size') or 1
        if len(images) < batch_size * n_iter:
            images += [images[0]] * (batch_size * n_iter - len(images))
        else:
            images = random.sample(images, batch_size * n_iter)

        local_images = []
        map = {}
        for key in images:
            if key not in map:
                map[key] = self._download_and_resize_img(key)
            local_images.append(map[key])

        return local_images

    def _download_and_resize_img(self, key: str) -> str:
        local = get_tmp_local_path(key)
        img = Image.open(local).convert("RGB")
        basename = os.path.basename(local)
        dirname = os.path.dirname(local)
        resize_path = os.path.join(dirname, "resize-" + basename)
        # img.resize((w, h)).save(resize_path)
        img.save(resize_path)
        img.close()

        return resize_path

    def _build_i2i_tasks(self, t: Task):
        tasks = []
        t['prompt'] = "(((best quality))),(((ultra detailed))), " + t['prompt']
        t[
            'negative_prompt'] = "(worst quality:2), (low quality:2), (normal quality:2), nude, (badhandv4:1.2), (easynegative), verybadimagenegative_v1.3, deformation, blurry, " + \
                                 t['negative_prompt']

        denoising_strengths = self._denoising_strengths(t)
        init_images = self._get_init_images(t)

        for i, denoising_strength in enumerate(denoising_strengths):
            t['denoising_strength'] = 0.1
            t['n_iter'] = 1
            t['batch_size'] = 1
            t["init_img"] = init_images[i] if len(init_images) > i else init_images[0]
            t['alwayson_scripts'] = {
                ADetailer: {
                    'args': [{
                        'ad_model': 'face_yolov8n_v2.pt',
                        'ad_mask_blur': 4,
                        'ad_denoising_strength': denoising_strength,
                        'ad_inpaint_only_masked': True,
                        'ad_inpaint_only_masked_padding': 64
                    }]
                }
            }
            tasks.append(Img2ImgTask.from_task(t, self.default_script_args))

        return tasks

    # def _build_t2i_tasks(self, t: Task):
    #     tasks = []
    #     t['prompt'] = "(((best quality))),(((ultra detailed))), " + t['prompt']
    #     t['negative_prompt'] = "(worst quality:2), (low quality:2), (normal quality:2), " + t['negative_prompt']

    #     denoising_strengths = self._denoising_strengths(t)
    #     init_images = self._get_init_images(t)
    #     for i, denoising_strength in enumerate(denoising_strengths):
    #         t['denoising_strength'] = 0.1
    #         t['n_iter'] = 1
    #         t['batch_size'] = 1
    #         init_img = init_images[i] if len(init_images) > i else init_images[0]
    #         t['alwayson_scripts'] = {
    #             ADetailer: {
    #                 'args': [{
    #                     'ad_model': 'face_yolov8n_v2.pt',
    #                     'ad_mask_blur': 4,
    #                     'ad_denoising_strength': denoising_strength,
    #                     'ad_inpaint_only_masked': True,
    #                     'ad_inpaint_only_masked_padding': 64
    #                 }]
    #             },
    #             "ControlNet": {
    #                 "args": [
    #                     {
    #                         "control_mode": "Balanced",
    #                         "enabled": True,
    #                         "guess_mode": False,
    #                         "guidance_end": 1,
    #                         "guidance_start": 0,
    #                         "image": {
    #                             "image": init_img,
    #                             "mask": ""
    #                         },
    #                         "invert_image": False,
    #                         "isShowModel": True,
    #                         "low_vram": False,
    #                         "model": "control_v11p_sd15_inpaint [ebff9138]",
    #                         "module": "inpaint_only",
    #                         "pixel_perfect": False,
    #                         "processor_res": 512,
    #                         "resize_mode": "Scale to Fit (Inner Fit)",
    #                         "tempImg": None,
    #                         "tempMask": None,
    #                         "threshold_a": 64,
    #                         "threshold_b": 64,
    #                         "weight": 1
    #                     }
    #                 ]
    #             }
    #         }
    #         tasks.append(Txt2ImgTask.from_task(t, self.default_script_args))

    #     return tasks

    # 白色背景
    def _get_face_mask(self, image, model_path):

        # 加载人脸关键点检测器
        face_model = os.path.join(model_path, r"face_detect/shape_predictor_68_face_landmarks.dat")
        if not os.path.isfile(face_model):
            import requests
            print("download shape_predictor_68_face_landmarks.dat from xingzheassert.obs.cn-north-4.myhuaweicloud.com")
            resp = requests.get(
                'https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/face/shape_predictor_68_face_landmarks.dat')
            if resp:
                dirname = os.path.dirname(face_model)
                os.makedirs(dirname, exist_ok=True)
                filepath = os.path.join("tmp", 'shape_predictor_68_face_landmarks.dat')
                os.makedirs('tmp', exist_ok=True)
                chunk_size = 1024

                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

                if os.path.isfile(filepath):
                    shutil.move(filepath, face_model)

        if not os.path.isfile(face_model):
            raise OSError(f'cannot found model:{face_model}')
        predictor = dlib.shape_predictor(face_model)

        image_tmp_local_path = get_tmp_local_path(image)
        image = Image.open(image_tmp_local_path)
        image = np.array(image.convert("RGB"))

        if not isinstance(image, (Image.Image, np.ndarray)):
            print("底图有错误")
            return None

        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用人脸检测器检测人脸
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # 创建一个与原始图像相同大小的掩膜
        mask = np.zeros_like(image)

        # 遍历检测到的人脸
        # print(len(faces))
        if len(faces) != 1:
            print(f"cannot detect face:{f}, result:{len(faces)}")
            return None

        white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)

        for face in faces:
            # 检测人脸关键点
            landmarks = predictor(gray, face)

            # 提取人脸轮廓
            points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append((x, y))

        # 计算所有点的最小和最大坐标值
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        # 将矩形区域设置为黑色

        cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), (255, 255, 255), -1)

        # 将蒙版应用到原始图像上
        # 将mask应用于原始图像
        result = cv2.bitwise_and(white_background, mask)

        return result

    def _get_multi_face_mask(self, image, model_path):

        # 加载人脸关键点检测器
        face_model = os.path.join(model_path, r"face_detect/shape_predictor_68_face_landmarks.dat")
        if not os.path.isfile(face_model):
            import requests
            print("download shape_predictor_68_face_landmarks.dat from xingzheassert.obs.cn-north-4.myhuaweicloud.com")
            resp = requests.get(
                'https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/resource/face/shape_predictor_68_face_landmarks.dat')
            if resp:
                dirname = os.path.dirname(face_model)
                os.makedirs(dirname, exist_ok=True)
                filepath = os.path.join("tmp", 'shape_predictor_68_face_landmarks.dat')
                os.makedirs('tmp', exist_ok=True)
                chunk_size = 1024

                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

                if os.path.isfile(filepath):
                    shutil.move(filepath, face_model)

        if not os.path.isfile(face_model):
            raise OSError(f'cannot found model:{face_model}')
        predictor = dlib.shape_predictor(face_model)

        # image_tmp_local_path = get_tmp_local_path(image)
        # image = Image.open(image_tmp_local_path)
        image = np.array(image.convert("RGB"))

        if not isinstance(image, (Image.Image, np.ndarray)):
            print("底图有错误")
            return None

        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用人脸检测器检测人脸
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # 创建一个与原始图像相同大小的掩膜
        mask = np.zeros_like(image)

        # 遍历检测到的人脸
        # print(len(faces))
        if len(faces) != 2:
            print(f"cannot detect face:{f}, result:{len(faces)}")
            return None

        # white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)

        result = {}
        result["boxes"] = []
        for face in faces:
            # 检测人脸关键点
            landmarks = predictor(gray, face)

            # 提取人脸轮廓
            points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append((x, y))

            # 计算所有点的最小和最大坐标值
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)
            # 将矩形区域设置为黑色
            result["boxes"].append([min_x, min_y, max_x, max_y])
        # cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), (255, 255, 255), -1)

        # 将蒙版应用到原始图像上
        # 将mask应用于原始图像
        # result = cv2.bitwise_and(white_background, mask)

        return result


    @FuncExecTimeWrapper()
    def _get_image_masks(self, init_images: typing.Sequence[str]):

        def gen_image_mask(init_image: str):
            face_model_path = os.path.join(os.getcwd(), "models")

            init_img_mask = self._get_face_mask(init_image, face_model_path)
            init_img_mask_image = Image.fromarray(init_img_mask)
            dirname = mk_tmp_dir("digital_mask")
            init_img_mask_path = os.path.join(dirname, uuid.uuid4().hex + ".png")

            init_img_mask_image.save(init_img_mask_path)

            return init_img_mask_path

        r = {}
        for image in init_images:
            mask = r.get(image)
            if not mask:
                r[image] = gen_image_mask(image)
            else:
                print(f"image mask cached:{image}")

        return r

    def _build_t2i_tasks(self, t: Task, merge_task: bool = False):
        tasks = []
        base_prompt = "solo,(((best quality))),(((ultra detailed))),(((masterpiece))),Ultra High Definition," \
                      "Maximum Detail Display,Hyperdetail,Clear details,Amazing quality,Super details,Unbelievable," \
                      "HDR,16K,details,The most authentic,Glossy solid color,"
        terms = (x.strip() for x in str(t['prompt']).split(','))
        lora_prompt = [x for x in terms if x.startswith('<lora:')]
        t['prompt'] = base_prompt + ','.join(lora_prompt)
        t['negative_prompt'] = "nsfw, paintings, sketches, (worst quality:2), (low quality:2), lowers, " \
                               "normal quality, ((monochrome)), ((grayscale)), logo, word, character, lowres," \
                               "bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits,cropped," \
                               "wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, " \
                               "username, blurry, bad feet, lowres, bad anatomy, bad hands, text,error, " \
                               "missing fngers,extra digt ,fewer digits,cropped, wort quality ,low quality," \
                               "normal quality, jpeg artifacts,signature,watermark, username, blurry, " \
                               "bad feet,hand" + t['negative_prompt']

        denoising_strengths = self._denoising_strengths(t)
        init_images = self._get_init_images(t)
        init_image_masks = self._get_image_masks(init_images)

        def get_alwayson_scripts(init_img: str, init_img_mask_path: str, denoising_strength: float):
            return {
                ADetailer: {
                    'args': [
                        {
                            "enabled": True,
                            "ad_model": "face_yolov8n_v2.pt",
                            "ad_prompt": "",
                            "ad_negative_prompt": "",
                            "ad_confidence": 0.3,
                            "ad_dilate_erode": 4,
                            "ad_mask_merge_invert": "None",
                            "ad_mask_blur": 4,
                            "ad_denoising_strength": denoising_strength,
                            "ad_inpaint_only_masked": True,
                            "ad_inpaint_only_masked_padding": 64
                        },
                        {
                            "enabled": True,
                            "ad_model": "face_yolov8n_v2.pt",
                            "ad_prompt": "",
                            "ad_negative_prompt": "",
                            "ad_confidence": 0.3,
                            "ad_dilate_erode": 4,
                            "ad_mask_merge_invert": "None",
                            "ad_mask_blur": 4,
                            "ad_denoising_strength": denoising_strength - 0.1,
                            "ad_inpaint_only_masked": True,
                            "ad_inpaint_only_masked_padding": 64
                        },
                    ]
                },
                "ControlNet": {
                    "args": [
                        {
                            "control_mode": "ControlNet is more important",
                            "enabled": True,
                            "guess_mode": False,
                            "guidance_end": 1,
                            "guidance_start": 0,
                            "image": {
                                "image": init_img,
                                "mask": ""
                            },
                            "invert_image": False,
                            "isShowModel": True,
                            "low_vram": False,
                            "model": "control_v11p_sd15_canny",
                            "module": "canny",
                            "pixel_perfect": True,
                            "processor_res": 512,
                            "resize_mode": "Crop and Resize",
                            "tempMask": "",
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "weight": 0.35
                        },
                        # {
                        #     "control_mode": "ControlNet is more important",
                        #     "enabled": True,
                        #     "guess_mode": False,
                        #     "guidance_end": 1,
                        #     "guidance_start": 0,
                        #     "image": {
                        #         "image": init_img,
                        #         "mask": ""
                        #     },
                        #     "invert_image": False,
                        #     "isShowModel": True,
                        #     "low_vram": False,
                        #     "model": "control_v11p_sd15_openpose",
                        #     "module": "openpose_faceonly",
                        #     "pixel_perfect": True,
                        #     "processor_res": 512,
                        #     "resize_mode": "Crop and Resize",
                        #     "tempMask": "",
                        #     "threshold_a": 64,
                        #     "threshold_b": 64,
                        #     "weight": 0.35
                        # },
                        {
                            "control_mode": "Balanced",
                            "enabled": True,
                            "guess_mode": False,
                            "guidance_end": 1,
                            "guidance_start": 0,
                            "image": {
                                "image": init_img,
                                "mask": init_img_mask_path
                            },
                            "invert_image": False,
                            "isShowModel": True,
                            "low_vram": False,
                            "model": "control_v11p_sd15_inpaint",
                            "module": "inpaint_only",
                            "pixel_perfect": True,
                            "processor_res": 512,
                            "resize_mode": "Crop and Resize",
                            "tempMask": "",
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "weight": 1
                        },
                    ]
                }
            }

        if not merge_task:
            for i, denoising_strength in enumerate(denoising_strengths):
                t['denoising_strength'] = 0.1
                t['n_iter'] = 1
                t['batch_size'] = 1

                init_img = init_images[i] if len(init_images) > i else init_images[0]
                init_img_mask_path = init_image_masks[init_img]

                t['alwayson_scripts'] = get_alwayson_scripts(init_img, init_img_mask_path, denoising_strength)
                tasks.append(Txt2ImgTask.from_task(t, self.default_script_args))
        else:
            init_image_count = defaultdict(int)
            for image_path in init_images:
                init_image_count[image_path] += 1
            for init_img, count in init_image_count.items():
                i = init_images.index(init_img) + count - 1
                denoising_strength = denoising_strengths[i]
                t['denoising_strength'] = denoising_strength
                t['n_iter'] = count
                t['batch_size'] = 1

                init_img = init_images[i] if len(init_images) > i else init_images[0]
                init_img_mask_path = init_image_masks[init_img]

                # img = Image.open(init_img)
                # t['width'] = 512
                # t['height'] = 768 if img.height % 768 == 0 else 512  # 仅支持512和768两个分辨率
                # img.close()

                t['alwayson_scripts'] = get_alwayson_scripts(init_img, init_img_mask_path, denoising_strength)
                tasks.append(Txt2ImgTask.from_task(t, self.default_script_args))

        return tasks

    # def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
    #     if task.minor_type == DigitalTaskType.Img2Img:
    #         yield from self._exec_img2img(task)
    #     elif task.minor_type == DigitalTaskType.Txt2Img:
    #         yield from self._exec_txt2img(task)

    def _set_img_local_path(self, img) -> str:
        tmp_path = "tmp"
        random_name = str(uuid.uuid4())[:8]
        img_path = os.path.join(tmp_path, "human-" + random_name + ".png")
        # img.resize((w, h)).save(resize_path)
        img.save(img_path)
        img.close()

        return img_path

    def _set_mask_local_path(self, img) -> str:
        tmp_path = "tmp"
        random_name = str(uuid.uuid4())[:8]
        mask_path = os.path.join(tmp_path, "mask-" + random_name + ".png")
        face_model_path = os.path.join(os.getcwd(), "models")
        mask = self._get_face_mask(img, face_model_path)
        cv2.imwrite(mask_path, mask)
        return mask_path

    def _build_multi_t2i_tasks(self, t: Task, first_lora_train_dataset, second_lora_train_dataset):
        tasks = []
        lora_models = t["lora_models"]
        base_prompt = "solo,(((best quality))),(((ultra detailed))),(((masterpiece))),Ultra High Definition,Maximum Detail Display,Hyperdetail,Clear details,Amazing quality,Super details,Unbelievable,HDR,16K,details,The most authentic,Glossy solid color,(((realistic))),looking at viewer fine art photography style,close up,portrait composition,elephoto lens,photograph,HDR,"
        terms = (x.strip() for x in str(t['prompt']).split(','))
        # lora_prompt = [x for x in terms if x.startswith('<lora:')]
        # t['prompt'] = base_prompt + ','.join(lora_prompt)
        t['prompt'] = base_prompt
        t[
            'negative_prompt'] = "nsfw, paintings, sketches, (worst quality:2), (low quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, lowres, bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits,cropped, wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet​, ​lowres, bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits,cropped, wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet,hand,"

        denoising_strengths = self._denoising_strengths(t)
        init_images = self._get_init_images(t)
        # init_image_masks = self._get_face_mask_double(init_images)

        print("len(init_images):", len(init_images))

        result = []
        for init_image in init_images:
            init_image_pil = Image.open(init_image)

            # face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd',
            #                           model_revision='v1.1')
            # result_det = face_detection(init_image_pil)
            face_model_path = os.path.join(os.getcwd(), "models")
            result_det = self._get_multi_face_mask(init_image_pil, face_model_path)
            bboxes = result_det['boxes']
            # assert(len(bboxes)) == 2
            bboxes = np.array(bboxes).astype(np.int16)
            lefts = []
            for bbox in bboxes:
                lefts.append(bbox[0])
            idxs = np.argsort(lefts)

            face_box = bboxes[idxs[0]]
            # inpaint_img_large = cv2.imread(self.inpaint_img)
            inpaint_img_large = cv2.cvtColor(np.array(init_image_pil), cv2.COLOR_RGB2BGR)
            mask_large = np.ones_like(inpaint_img_large)
            mask_large1 = np.zeros_like(inpaint_img_large)
            h, w, _ = inpaint_img_large.shape
            for i in range(len(bboxes)):
                if i != idxs[0]:
                    bbox = bboxes[i]
                    inpaint_img_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
                    mask_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            face_ratio = 0.45
            cropl = int(max(face_box[3] - face_box[1], face_box[2] - face_box[0]) / face_ratio / 2)
            cx = int((face_box[2] + face_box[0]) / 2)
            cy = int((face_box[1] + face_box[3]) / 2)
            cropup = min(cy, cropl)
            cropbo = min(h - cy, cropl)
            crople = min(cx, cropl)
            cropri = min(w - cx, cropl)
            inpaint_img = np.pad(inpaint_img_large[cy - cropup:cy + cropbo, cx - crople:cx + cropri],
                                 ((cropl - cropup, cropl - cropbo), (cropl - crople, cropl - cropri), (0, 0)),
                                 'constant')
            inpaint_img = cv2.resize(inpaint_img, (512, 512))
            inpaint_img = Image.fromarray(inpaint_img[:, :, ::-1])
            mask_large1[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = 1
            mask_large = mask_large * mask_large1

            human_img_path1 = self._set_img_local_path(inpaint_img)
            human_mask_path1 = self._set_mask_local_path(human_img_path1)

            def get_alwayson_scripts(init_img: str, init_img_mask_path: str, denoising_strength: float):
                return {
                    ADetailer: {
                        'args': [
                            {
                                "enabled": True,
                                "ad_model": "face_yolov8n_v2.pt",
                                "ad_prompt": "",
                                "ad_negative_prompt": "",
                                "ad_confidence": 0.3,
                                "ad_dilate_erode": 4,
                                "ad_mask_merge_invert": "None",
                                "ad_mask_blur": 4,
                                "ad_denoising_strength": denoising_strength,
                                "ad_inpaint_only_masked": True,
                                "ad_inpaint_only_masked_padding": 64
                            },
                            {
                                "enabled": True,
                                "ad_model": "face_yolov8n_v2.pt",
                                "ad_prompt": "",
                                "ad_negative_prompt": "",
                                "ad_confidence": 0.3,
                                "ad_dilate_erode": 4,
                                "ad_mask_merge_invert": "None",
                                "ad_mask_blur": 4,
                                "ad_denoising_strength": denoising_strength - 0.1,
                                "ad_inpaint_only_masked": True,
                                "ad_inpaint_only_masked_padding": 64
                            },
                        ]
                    },
                    "ControlNet": {
                        "args": [
                            {
                                "control_mode": "ControlNet is more important",
                                "enabled": True,
                                "guess_mode": False,
                                "guidance_end": 1,
                                "guidance_start": 0,
                                "image": {
                                    "image": init_img,
                                    "mask": ""
                                },
                                "invert_image": False,
                                "isShowModel": True,
                                "low_vram": False,
                                "model": "control_v11p_sd15_canny",
                                "module": "canny",
                                "pixel_perfect": True,
                                "processor_res": 512,
                                "resize_mode": "Crop and Resize",
                                "tempMask": "",
                                "threshold_a": 64,
                                "threshold_b": 64,
                                "weight": 0.35
                            },
                            {
                                "control_mode": "ControlNet is more important",
                                "enabled": True,
                                "guess_mode": False,
                                "guidance_end": 1,
                                "guidance_start": 0,
                                "image": {
                                    "image": init_img,
                                    "mask": ""
                                },
                                "invert_image": False,
                                "isShowModel": True,
                                "low_vram": False,
                                "model": "control_v11p_sd15_openpose",
                                "module": "openpose_faceonly",
                                "pixel_perfect": True,
                                "processor_res": 512,
                                "resize_mode": "Crop and Resize",
                                "tempMask": "",
                                "threshold_a": 64,
                                "threshold_b": 64,
                                "weight": 0.35
                            },
                            {
                                "control_mode": "Balanced",
                                "enabled": True,
                                "guess_mode": False,
                                "guidance_end": 1,
                                "guidance_start": 0,
                                "image": {
                                    "image": init_img,
                                    "mask": init_img_mask_path
                                },
                                "invert_image": False,
                                "isShowModel": True,
                                "low_vram": False,
                                "model": "control_v11p_sd15_inpaint",
                                "module": "inpaint_only",
                                "pixel_perfect": True,
                                "processor_res": 512,
                                "resize_mode": "Crop and Resize",
                                "tempMask": "",
                                "threshold_a": 64,
                                "threshold_b": 64,
                                "weight": 1
                            },
                        ]
                    }
                }

            # # select_high_quality_face PIL
            # selected_face = select_high_quality_face(input_img_dir1)
            # # face_swap cv2
            # swap_results = face_swap_fn(self.use_face_swap, gen_results, selected_face)
            # # stylization
            # final_gen_results = swap_results

            # if not merge_task:

            # for i, denoising_strength in enumerate(denoising_strengths):
            denoising_strength = 0.5
            t['denoising_strength'] = 0.1
            t['n_iter'] = 1
            t['batch_size'] = 1

            init_img = human_img_path1
            init_img_mask_path = human_mask_path1

            print("init_img, init_img_mask_path:", init_img, init_img_mask_path)
            t['alwayson_scripts'] = get_alwayson_scripts(init_img, init_img_mask_path, denoising_strength)
            # tasks.append(Txt2ImgTask.from_task(t, self.default_script_args))
            # else:
            #     init_image_count = defaultdict(int)
            #     for image_path in init_images:
            #         init_image_count[image_path] += 1
            #     for init_img, count in init_image_count.items():
            #         i = init_images.index(init_img) + count - 1
            #         denoising_strength = denoising_strengths[i]
            #         t['denoising_strength'] = denoising_strength
            #         t['n_iter'] = count
            #         t['batch_size'] = 1

            #         init_img = init_images[i] if len(init_images) > i else init_images[0]
            #         init_img_mask_path = init_image_masks[init_img]

            #         # img = Image.open(init_img)
            #         # t['width'] = 512
            #         # t['height'] = 768 if img.height % 768 == 0 else 512  # 仅支持512和768两个分辨率
            #         # img.close()

            #         t['alwayson_scripts'] = get_alwayson_scripts(init_img, init_img_mask_path, denoising_strength)
            #         # tasks.append(Txt2ImgTask.from_task(t, self.default_script_args))
            # t["lora_models"] = lora_models[0]
            parts = lora_models[0].split('/')
            lora_name = parts[-1].split('.')[0]
            t["prompt"] = base_prompt + "<lora:" + lora_name + ":0.9>"
            task = Txt2ImgTask.from_task(t, self.default_script_args)
            task.do_not_save_grid = True
            processed = process_images(task)
            gen_results = processed.images[0]

            input_img_dir1 = first_lora_train_dataset

            if input_img_dir1 is not None:
                # select_high_quality_face PIL
                selected_face = select_high_quality_face(input_img_dir1)
                # face_swap cv2
                swap_results = face_swap_fn(True, [gen_results], selected_face)
                # stylization
                final_gen_results = swap_results

            else:
                final_gen_results = [cv2.cvtColor(np.array(gen_results), cv2.COLOR_RGB2BGR)]
            # print(len(final_gen_results))

            final_gen_results_new = []
            # inpaint_img_large = cv2.imread(self.inpaint_img)
            inpaint_img_large = cv2.cvtColor(np.array(init_image_pil), cv2.COLOR_RGB2BGR)
            ksize = int(10 * cropl / 256)
            # for i in range(len(final_gen_results)):

            print('Start cropping.')
            # rst_gen = cv2.resize(final_gen_results, (cropl * 2, cropl * 2))
            rst_gen = cv2.resize(final_gen_results[0], (cropl * 2, cropl * 2))
            rst_crop = rst_gen[cropl - cropup:cropl + cropbo, cropl - crople:cropl + cropri]
            print(rst_crop.shape)
            inpaint_img_rst = np.zeros_like(inpaint_img_large)
            print('Start pasting.')
            print("第一次paste,inpaint_img_rst的形状,inpaint_img_rst.shape, rst_crop.shape:", inpaint_img_rst.shape,
                  rst_crop.shape)
            inpaint_img_rst[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = rst_crop
            print('Fininsh pasting.')
            print(inpaint_img_rst.shape, mask_large.shape, inpaint_img_large.shape)
            mask_large = mask_large.astype(np.float32)
            kernel = np.ones((ksize * 2, ksize * 2))
            mask_large1 = cv2.erode(mask_large, kernel, iterations=1)
            mask_large1 = cv2.GaussianBlur(mask_large1, (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
            mask_large1[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 1
            mask_large = mask_large * mask_large1
            final_inpaint_rst = (
                    inpaint_img_rst.astype(np.float32) * mask_large.astype(np.float32) + inpaint_img_large.astype(
                np.float32) * (1.0 - mask_large.astype(np.float32))).astype(np.uint8)
            print('Finish masking.')
            final_gen_results_new.append(final_inpaint_rst)
            print('Finish generating.')

        face_box = bboxes[idxs[1]]
        mask_large = np.ones_like(inpaint_img_large)
        mask_large1 = np.zeros_like(inpaint_img_large)
        h, w, _ = inpaint_img_large.shape
        for i in range(len(bboxes)):
            if i != idxs[1]:
                bbox = bboxes[i]
                inpaint_img_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
                mask_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        face_ratio = 0.45
        cropl = int(max(face_box[3] - face_box[1], face_box[2] - face_box[0]) / face_ratio / 2)
        cx = int((face_box[2] + face_box[0]) / 2)
        cy = int((face_box[1] + face_box[3]) / 2)
        cropup = min(cy, cropl)
        cropbo = min(h - cy, cropl)
        crople = min(cx, cropl)
        cropri = min(w - cx, cropl)
        mask_large1[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = 1
        mask_large = mask_large * mask_large1

        # inpaint_imgs = []
        # for i in range(1):
        # inpaint_img_large = final_gen_results_new[i] * mask_large
        inpaint_img = np.pad(inpaint_img_large[cy - cropup:cy + cropbo, cx - crople:cx + cropri],
                             ((cropl - cropup, cropl - cropbo), (cropl - crople, cropl - cropri), (0, 0)), 'constant')
        inpaint_img = cv2.resize(inpaint_img, (512, 512))
        inpaint_img = Image.fromarray(inpaint_img[:, :, ::-1])
        # inpaint_imgs.append(inpaint_img)

        human_img_path2 = self._set_img_local_path(inpaint_img)
        human_mask_path2 = self._set_mask_local_path(human_img_path2)

        def get_alwayson_scripts(init_img: str, init_img_mask_path: str, denoising_strength: float):
            return {
                ADetailer: {
                    'args': [
                        {
                            "enabled": True,
                            "ad_model": "face_yolov8n_v2.pt",
                            "ad_prompt": "",
                            "ad_negative_prompt": "",
                            "ad_confidence": 0.3,
                            "ad_dilate_erode": 4,
                            "ad_mask_merge_invert": "None",
                            "ad_mask_blur": 4,
                            "ad_denoising_strength": denoising_strength,
                            "ad_inpaint_only_masked": True,
                            "ad_inpaint_only_masked_padding": 64
                        },
                        {
                            "enabled": True,
                            "ad_model": "face_yolov8n_v2.pt",
                            "ad_prompt": "",
                            "ad_negative_prompt": "",
                            "ad_confidence": 0.3,
                            "ad_dilate_erode": 4,
                            "ad_mask_merge_invert": "None",
                            "ad_mask_blur": 4,
                            "ad_denoising_strength": denoising_strength - 0.1,
                            "ad_inpaint_only_masked": True,
                            "ad_inpaint_only_masked_padding": 64
                        },
                    ]
                },
                "ControlNet": {
                    "args": [
                        {
                            "control_mode": "ControlNet is more important",
                            "enabled": True,
                            "guess_mode": False,
                            "guidance_end": 1,
                            "guidance_start": 0,
                            "image": {
                                "image": init_img,
                                "mask": ""
                            },
                            "invert_image": False,
                            "isShowModel": True,
                            "low_vram": False,
                            "model": "control_v11p_sd15_canny",
                            "module": "canny",
                            "pixel_perfect": True,
                            "processor_res": 512,
                            "resize_mode": "Crop and Resize",
                            "tempMask": "",
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "weight": 0.35
                        },
                        {
                            "control_mode": "ControlNet is more important",
                            "enabled": True,
                            "guess_mode": False,
                            "guidance_end": 1,
                            "guidance_start": 0,
                            "image": {
                                "image": init_img,
                                "mask": ""
                            },
                            "invert_image": False,
                            "isShowModel": True,
                            "low_vram": False,
                            "model": "control_v11p_sd15_openpose",
                            "module": "openpose_faceonly",
                            "pixel_perfect": True,
                            "processor_res": 512,
                            "resize_mode": "Crop and Resize",
                            "tempMask": "",
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "weight": 0.35
                        },
                        {
                            "control_mode": "Balanced",
                            "enabled": True,
                            "guess_mode": False,
                            "guidance_end": 1,
                            "guidance_start": 0,
                            "image": {
                                "image": init_img,
                                "mask": init_img_mask_path
                            },
                            "invert_image": False,
                            "isShowModel": True,
                            "low_vram": False,
                            "model": "control_v11p_sd15_inpaint",
                            "module": "inpaint_only",
                            "pixel_perfect": True,
                            "processor_res": 512,
                            "resize_mode": "Crop and Resize",
                            "tempMask": "",
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "weight": 1
                        },
                    ]
                }
            }

        # for i, denoising_strength in enumerate(denoising_strengths):
        denoising_strength = 0.5
        t['denoising_strength'] = 0.1
        t['n_iter'] = 1
        t['batch_size'] = 1

        init_img = human_img_path2
        init_img_mask_path = human_mask_path2

        print("init_img, init_img_mask_path:", init_img, init_img_mask_path)
        t['alwayson_scripts'] = get_alwayson_scripts(init_img, init_img_mask_path, denoising_strength)

        # t["lora_models"] = lora_models[1]
        parts = lora_models[1].split('/')
        lora_name = parts[-1].split('.')[0]
        t["prompt"] = base_prompt + "<lora:" + lora_name + ":0.9>"
        task = Txt2ImgTask.from_task(t, self.default_script_args)
        task.do_not_save_grid = True
        processed = process_images(task)
        gen_results = processed.images[0]

        input_img_dir2 = second_lora_train_dataset

        if input_img_dir2 is not None:
            selected_face = select_high_quality_face(input_img_dir2)
            # face_swap cv2
            swap_results = face_swap_fn(True, [gen_results], selected_face)
            # stylization
            final_gen_results = swap_results
        else:

            # stylization
            final_gen_results = [cv2.cvtColor(np.array(gen_results), cv2.COLOR_RGB2BGR)]
        # print(len(final_gen_results))

        final_gen_results_final = []
        # inpaint_img_large = cv2.imread(self.inpaint_img)
        inpaint_img_large = cv2.cvtColor(np.array(init_image_pil), cv2.COLOR_RGB2BGR)
        ksize = int(10 * cropl / 256)
        print('Start cropping.')
        # rst_gen = cv2.resize(final_gen_results, (cropl * 2, cropl * 2))
        rst_gen = cv2.resize(final_gen_results[0], (cropl * 2, cropl * 2))
        rst_crop = rst_gen[cropl - cropup:cropl + cropbo, cropl - crople:cropl + cropri]
        print(rst_crop.shape)
        inpaint_img_rst = np.zeros_like(inpaint_img_large)
        print('Start pasting.')
        print("第二次paste,inpaint_img_rst的形状,inpaint_img_rst.shape, rst_crop.shape:", inpaint_img_rst.shape,
              rst_crop.shape)
        inpaint_img_rst[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = rst_crop
        print('Fininsh pasting.')
        print(inpaint_img_rst.shape, mask_large.shape, inpaint_img_large.shape)
        mask_large = mask_large.astype(np.float32)
        kernel = np.ones((ksize * 2, ksize * 2))
        mask_large1 = cv2.erode(mask_large, kernel, iterations=1)
        mask_large1 = cv2.GaussianBlur(mask_large1, (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
        mask_large1[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 1
        mask_large = mask_large * mask_large1
        final_inpaint_rst = (inpaint_img_rst.astype(np.float32) * mask_large.astype(np.float32) + final_gen_results_new[
            0].astype(np.float32) * (1.0 - mask_large.astype(np.float32))).astype(np.uint8)
        print('Finish masking.')
        final_gen_results_final.append(final_inpaint_rst)
        print('Finish generating.')

        processed.images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in
                            final_gen_results_final]

        return final_gen_results_final, processed, task

    def _exec_img2img(self, task: Task) -> typing.Iterable[TaskProgress]:
        time_start = time.time()
        base_model_path = self._get_local_checkpoint(task)
        load_sd_model_weights(base_model_path, task.model_hash)
        progress = TaskProgress.new_ready(task, f'model loaded, gen refine image...', 50)
        self._refresh_default_script_args()
        yield progress
        tasks = self._build_i2i_tasks(task)
        # i2i
        images = []
        all_seeds = []
        all_subseeds = []
        processed = None
        upload_files_eta_secs = 5

        for i, p in enumerate(tasks):
            if i == 0:
                self._set_little_models(p)
            processed = process_images(p)
            all_seeds.extend(processed.all_seeds)
            all_subseeds.extend(processed.all_subseeds)
            images.append(processed.images[0])
            progress.task_progress = min((i + 1) * 100 / len(tasks), 98)
            # time_since_start = time.time() - time_start
            # eta = (time_since_start / p)
            # progress.eta_relative = int(eta - time_since_start) + upload_files_eta_secs
            if i == 0:
                progress.eta_relative = 60
            else:
                progress.calc_eta_relative(upload_files_eta_secs)
            yield progress
            p.close()

        # 开启宫格图
        if task.get('grid_enable', False):
            grid = modules.images.image_grid(images, len(images))
            images.insert(0, grid)
            processed.index_of_first_image = 1
            processed.index_of_end_image = len(images)
        else:
            processed.index_of_first_image = 0
            processed.index_of_end_image = len(images) - 1
        processed.images = images

        progress.status = TaskStatus.Uploading
        yield progress

        images = save_processed_images(processed,
                                       tasks[0].outpath_samples,
                                       tasks[0].outpath_grids,
                                       tasks[0].outpath_scripts,
                                       task.id,
                                       inspect=False,
                                       forbidden_review=True)

        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(all_seeds, all_subseeds)

        yield progress

    def _exec_txt2img(self, task: Task) -> typing.Iterable[TaskProgress]:
        time_start = time.time()
        base_model_path = self._get_local_checkpoint(task)
        load_sd_model_weights(base_model_path, task.model_hash)
        progress = TaskProgress.new_ready(task, f'model loaded, gen refine image...', 120)
        self._refresh_default_script_args()
        yield progress

        merge_task = True
        tasks = self._build_t2i_tasks(task, merge_task)
        # i2i
        images = []
        all_seeds = []
        all_subseeds = []
        processed = None
        upload_files_eta_secs = 20

        start_time = time.time()
        logger.info(f"preprocess digtal t2i cost:{time.time() - time_start}s")
        shared.state.begin()
        for i, p in enumerate(tasks):
            if i == 0:
                self._set_little_models(p)
            print(f"> process task n_iter:{p.n_iter}")

            def update_progress():
                image_numbers = p.n_iter * p.batch_size
                if image_numbers <= 0:
                    image_numbers = 1
                task_p = 0
                if shared.state.job_count > 0:
                    job_no = shared.state.job_no - 1 if shared.state.job_no > 0 else 0
                    task_p += job_no / (image_numbers)
                    # p += (shared.state.job_no) / shared.state.job_count
                elif shared.state.sampling_steps > 0:
                    task_p += 1 / (image_numbers) * shared.state.sampling_step / shared.state.sampling_steps
                cp = min((i + 1) * 100 * task_p / len(tasks) * 0.65, 98)
                if cp - progress.task_progress > 8:
                    progress.task_progress = int(cp)
                    if i == 0 and len(tasks) > 1:
                        progress.eta_relative = 90
                    else:
                        progress.calc_eta_relative(upload_files_eta_secs)
                    self._set_task_status(progress)

            shared.state.current_latent_changed_callback = update_progress
            p.do_not_save_grid = True
            processed = process_images(p)
            all_seeds.extend(processed.all_seeds)
            all_subseeds.extend(processed.all_subseeds)
            if not merge_task:
                images.append(processed.images[0])
            else:
                images.extend(processed.images[:processed.index_of_end_image + 1])

            # time_since_start = time.time() - time_start
            # eta = (time_since_start / p)
            # progress.eta_relative = int(eta - time_since_start) + upload_files_eta_secs
            # if i == 0:
            #     progress.eta_relative = 90
            # else:
            #     progress.calc_eta_relative(upload_files_eta_secs)
            # yield progress
            p.close()
        shared.state.end()
        logger.info(f"inference digtal t2i cost:{time.time() - start_time}s")
        # 开启宫格图
        if task.get('grid_enable', False):
            grid = modules.images.image_grid(images, len(images))
            images.insert(0, grid)
            processed.index_of_first_image = 1
            processed.index_of_end_image = len(images)
        else:
            processed.index_of_first_image = 0
            processed.index_of_end_image = len(images) - 1
        processed.images = images

        progress.status = TaskStatus.Uploading
        yield progress

        images = save_processed_images(processed,
                                       tasks[0].outpath_samples,
                                       tasks[0].outpath_grids,
                                       tasks[0].outpath_scripts,
                                       task.id,
                                       inspect=False,
                                       forbidden_review=True)

        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(all_seeds, all_subseeds)

        yield progress

    def _exec_multi_portrait(self, task: Task):
        self._refresh_default_script_args()
        print("self.default_script_args:", self.default_script_args)

        print("task:", task)

        print("type(task):", type(task))
        multi_process_task = MultiGenDigitalPhotoTask.from_task(task, self.default_script_args)
        # 加载底模
        base_model_path = self._get_local_checkpoint(task)
        load_sd_model_weights(base_model_path, task.model_hash)
        # 下载LORA
        progress = TaskProgress.new_ready(task, f'model loaded, gen refine image...', 120)
        self._set_little_models(multi_process_task)
        yield progress
        # 获取LORA训练图片
        train_image_dirs = multi_process_task.get_train_image_local_dir()
        # 第1个LORA图片所在目录
        first_lora_train_dataset = train_image_dirs[0]
        # 第2个LORA图片所在目录
        second_lora_train_dataset = train_image_dirs[1]

        final_image, processed, out_task = self._build_multi_t2i_tasks(task, first_lora_train_dataset,
                                                                       second_lora_train_dataset)

        progress.status = TaskStatus.Uploading
        yield progress
        all_seeds = []
        all_subseeds = []
        all_seeds.extend(processed.all_seeds)
        all_subseeds.extend(processed.all_subseeds)

        task['all_seed'] = all_seeds
        task['all_sub_seed'] = all_subseeds

        print("task:", task)
        print("out_task:", out_task)

        out_task.id = task["task_id"]
        images = save_processed_images(processed,
                                       out_task.outpath_samples,
                                       out_task.outpath_grids,
                                       out_task.outpath_scripts,
                                       out_task.id,
                                       inspect=False,
                                       forbidden_review=True)

        # # todo: 生图逻辑

        # images = {
        #     'samples': {
        #         'high': []  # 原图KEY
        #     }
        # }
        progress = TaskProgress.new_finish(task, images)
        yield progress

        progress.update_seed(all_seeds, all_subseeds)
        yield progress
