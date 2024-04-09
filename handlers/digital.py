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
from enum import IntEnum
from PIL import ImageOps, Image
from handlers.txt2img import Txt2ImgTask
from handlers.img2img import Img2ImgTask, Img2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, ADetailer, mk_tmp_dir
from handlers.utils import get_tmp_local_path, upload_files, upload_pil_image
from loguru import logger
from tools.wrapper import FuncExecTimeWrapper

class DigitalTaskType(IntEnum):
    Img2Img = 2  # 原本是1
    Txt2Img = 1  # 原本是2


# class DigitalTaskHandler(Txt2ImgTaskHandler):
class DigitalTaskHandler(Img2ImgTaskHandler):

    def __init__(self):
        super(DigitalTaskHandler, self).__init__()
        self.task_type = TaskType.Digital
        self.register(
            (DigitalTaskType.Img2Img, self._exec_img2img),
            (DigitalTaskType.Txt2Img, self._exec_txt2img)
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
            return [0.45, 0.45, 0.5, 0.5]
        return [0.45, 0.5, 0.5, 0.55]

    def _get_init_images(self, t: Task):
        images = (t.get('init_img') or "").split(',')
        n_iter = t.get('n_iter') or 1
        batch_size = t.get('batch_size') or 1
        if len(images) < batch_size * n_iter:
            images += [images[0]] * (batch_size * n_iter - len(images))
        else:
            images = random.sample(images, batch_size * n_iter)
        return images

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

    def _build_t2i_tasks(self, t: Task):
        tasks = []
        t[
            'prompt'] = "solo,(((best quality))),(((ultra detailed))),(((masterpiece))),Ultra High Definition,Maximum Detail Display,Hyperdetail,Clear details,Amazing quality,Super details,Unbelievable,HDR,16K,details,The most authentic,Glossy solid color," + \
                        t['prompt']
        t[
            'negative_prompt'] = "nsfw, paintings, sketches, (worst quality:2), (low quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, " + \
                                 t['negative_prompt']

        denoising_strengths = self._denoising_strengths(t)
        init_images = self._get_init_images(t)
        init_image_masks = self._get_image_masks(init_images)

        for i, denoising_strength in enumerate(denoising_strengths):
            t['denoising_strength'] = 0.1
            t['n_iter'] = 1
            t['batch_size'] = 1
            t.get("cfg_scale", 5)

            init_img = init_images[i] if len(init_images) > i else init_images[0]
            init_img_mask_path = init_image_masks[init_img]

            t['alwayson_scripts'] = {
                ADetailer: {
                    'args': [{
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
                        }, ]
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
            tasks.append(Txt2ImgTask.from_task(t, self.default_script_args))

        return tasks

    # def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
    #     if task.minor_type == DigitalTaskType.Img2Img:
    #         yield from self._exec_img2img(task)
    #     elif task.minor_type == DigitalTaskType.Txt2Img:
    #         yield from self._exec_txt2img(task)

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
        progress = TaskProgress.new_ready(task, f'model loaded, gen refine image...', 50)
        self._refresh_default_script_args()
        yield progress
        tasks = self._build_t2i_tasks(task)
        # i2i
        images = []
        all_seeds = []
        all_subseeds = []
        processed = None
        upload_files_eta_secs = 5

        start_time = time.time()
        logger.info(f"preprocess digtal t2i cost:{time.time() - time_start}s")

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
        logger.info(f"inference digtal t2i cost:{time.time()-start_time}s")
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