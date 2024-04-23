#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/19 2:43 PM
# @Author  : wangdongming
# @Site    : 
# @File    : multi_digital.py
# @Software: xingzhe.ai
import random
import typing
import uuid
import os
import shutil
import dlib
import cv2
import numpy as np
from PIL import ImageOps, Image
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import process_images
from handlers.utils import load_sd_model_weights, save_processed_images, ADetailer, mk_tmp_dir
from handlers.utils import get_tmp_local_path
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks





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

    def lora_prompts(self, w=0.9):
        prompts = []
        for i in range(len(self.lora_meta_map)):
            lora_key = self.lora_meta_map[i].lora_key
            name, _ = os.path.splitext(os.path.basename(lora_key))
            prompts.append(f'<lora:{name}:{w}>')
        return prompts

    @classmethod
    def from_task(cls, task: Task, default_script_args: typing.Sequence, refiner_checkpoint: str = None):
        base_model_path = task['base_model_path']
        alwayson_scripts = task['alwayson_scripts']
        user_id = task['user_id']
        select_script = task.get('select_script')
        select_script_name, select_script_args = None, None
        prompt = task.get('prompt', '')
        negative_prompt = task.get('negative_prompt', '')
        lora_meta_array = task.get('lora_meta_array')

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
        if 'lora_meta_array' in kwargs:
            kwargs.pop('lora_meta_array')

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

        print("lora_meta_array:", lora_meta_array)
        return cls(base_model_path,
                   user_id,
                   lora_meta_array,
                   default_script_args,
                   prompt=prompt,
                   negative_prompt=negative_prompt,
                   alwayson_scripts=alwayson_scripts,
                   select_script_name=select_script_name,
                   select_script_args=select_script_args,
                    # default_script_arg_txt2img = default_script_args,
                   **kwargs)


class MultiGenPortraitHandler(Txt2ImgTaskHandler):

    def get_alwayson_scripts(self, init_img: str, init_img_mask_path: str, denoising_strength: float=0.5):
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

    def _set_img_local_path(self, img) -> str:
        tmp_path = "tmp"
        random_name = str(uuid.uuid4())[:8]
        img_path = os.path.join(tmp_path, "human-" + random_name + ".png")
        # img.resize((w, h)).save(resize_path)
        img.save(img_path)
        img.close()

        return img_path

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

    def _set_mask_local_path(self, img) -> str:
        tmp_path = "tmp"
        random_name = str(uuid.uuid4())[:8]
        mask_path = os.path.join(tmp_path, "mask-" + random_name + ".png")
        face_model_path = os.path.join(os.getcwd(), "models")
        mask = self._get_face_mask(img, face_model_path)
        cv2.imwrite(mask_path, mask)
        return mask_path

    def select_high_quality_face(self, input_img_dir):
        input_img_dir = str(input_img_dir)
        quality_score_list = []
        abs_img_path_list = []

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

    def segment(self, segmentation_pipeline, img, ksize=0.1, eyeh=0, ksize1=0, include_neck=False, warp_mask=None,
                return_human=False):
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
        # mask_hair = np.clip(mask_hair, 0, 1)
        mask_neck = np.clip(mask_neck, 0, 1)
        # mask_cloth = np.clip(mask_cloth, 0, 1)
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
                raise ValueError("soft_mask undefine")
                # kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                # kernel1 = np.ones((kernel_size1, kernel_size1))
                # soft_mask = cv2.dilate(mask_face, kernel1, iterations=1)
            else:
                soft_mask = mask_face
        if include_neck:
            soft_mask = np.clip(soft_mask + mask_neck, 0, 1)

        if return_human:
            mask_human = cv2.GaussianBlur(mask_human, (21, 21), 0) * mask_human
            return soft_mask, mask_human
        else:
            return soft_mask

    def face_swap_fn(self, use_face_swap, gen_results, template_face):
        if use_face_swap:

            out_img_list = []
            image_face_fusion = pipeline('face_fusion_torch',
                                         model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.5')
            segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
            for img in gen_results:
                result = image_face_fusion(dict(template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
                face_mask = self.segment(segmentation_pipeline, img, ksize=0.1)
                result = (result * face_mask[:, :, None] + np.array(img)[:, :, ::-1] * (
                            1 - face_mask[:, :, None])).astype(
                    np.uint8)
                out_img_list.append(result)
            return out_img_list
        else:
            ret_results = []
            for img in gen_results:
                ret_results.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            return ret_results

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
                local = get_tmp_local_path(key)
                if local:
                    map[key] = local

            local_images.append(map[key])

        return local_images

    def gen_portraits(self, t: Task):

        base_prompt = "solo,(((best quality))),(((ultra detailed))),(((masterpiece))),Ultra High Definition," \
                      "Maximum Detail Display,Hyperdetail,Clear details,Amazing quality,Super details," \
                      "Unbelievable,HDR,16K,details,The most authentic,Glossy solid color,(((realistic)))," \
                      "looking at viewer fine art photography style,close up,portrait composition," \
                      "elephoto lens,photograph,HDR,"

        t['negative_prompt'] = "nsfw, paintings, sketches, (worst quality:2), (low quality:2), lowers, normal quality," \
                               " ((monochrome)), ((grayscale)), logo, word, character, lowres, bad anatomy, bad hands, " \
                               "text,error, missing fngers,extra digt ,fewer digits,cropped, wort quality ,low quality," \
                               "normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet, " \
                               "lowres, bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits," \
                               "cropped, wort quality ,low quality,normal quality, jpeg artifacts,signature," \
                               "watermark, username, blurry, bad feet,hand,"
        self._refresh_default_script_args()
        t['denoising_strength'] = 0.1
        t['n_iter'] = 1
        t['batch_size'] = 1

        process_args = MultiGenDigitalPhotoTask.from_task(t, self.default_script_args)
        lora_prompts = process_args.lora_prompts()
        # 加载底模
        base_model_path = self._get_local_checkpoint(t)
        load_sd_model_weights(base_model_path, t.model_hash)
        # 下载LORA
        progress = TaskProgress.new_ready(t, f'model loaded, gen refine image...', 300)
        self._set_little_models(process_args)
        # 下载图片
        init_images = self._get_init_images(t)
        init_image = init_images[0]
        init_image_pil = Image.open(init_image)
        # 检测人脸
        # face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd',
        #                           model_revision='v1.1')
        # result_det = face_detection(init_image_pil)
        face_model_path = os.path.join(os.getcwd(), "models")
        result_det = self._get_multi_face_mask(init_image_pil, face_model_path)
        bboxes = result_det['boxes']
        # assert(len(bboxes)) == 2
        bboxes = np.array(bboxes).astype(np.int16)
        yield progress

        # 获取LORA训练图片
        train_image_dirs = process_args.get_train_image_local_dir()
        # 第1个LORA图片所在目录
        first_lora_train_dataset = train_image_dirs[0]
        # 第2个LORA图片所在目录
        second_lora_train_dataset = train_image_dirs[1]

        upload_files_eta_secs = 5
        # ================================================================
        # ========================> 第一次生成  <==========================
        # ================================================================
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

        t['alwayson_scripts'] = self.get_alwayson_scripts(human_img_path1, human_mask_path1)
        process_args1 = MultiGenDigitalPhotoTask.from_task(t, self.default_script_args)
        process_args1.prompt = base_prompt + f",{lora_prompts[0]}"
        process_args1.do_not_save_grid = True
        processed = process_images(process_args1)
        gen_results = processed.images[0]

        progress.task_progress = 20
        progress.calc_eta_relative(upload_files_eta_secs)
        yield progress

        if first_lora_train_dataset:
            # select_high_quality_face PIL
            selected_face = self.select_high_quality_face(first_lora_train_dataset)
            # face_swap cv2
            swap_results = self.face_swap_fn(True, [gen_results], selected_face)
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

        progress.task_progress = 45
        progress.calc_eta_relative(upload_files_eta_secs)
        yield progress
        # ================================================================
        # ========================> 第二次生成  <==========================
        # ================================================================
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

        inpaint_img = np.pad(inpaint_img_large[cy - cropup:cy + cropbo, cx - crople:cx + cropri],
                             ((cropl - cropup, cropl - cropbo), (cropl - crople, cropl - cropri), (0, 0)), 'constant')
        inpaint_img = cv2.resize(inpaint_img, (512, 512))
        inpaint_img = Image.fromarray(inpaint_img[:, :, ::-1])
        # inpaint_imgs.append(inpaint_img)

        human_img_path2 = self._set_img_local_path(inpaint_img)
        human_mask_path2 = self._set_mask_local_path(human_img_path2)

        t['alwayson_scripts'] = self.get_alwayson_scripts(human_img_path2, human_mask_path2)
        process_args2 = MultiGenDigitalPhotoTask.from_task(t, self.default_script_args)
        process_args2.do_not_save_grid = True
        process_args2.prompt = base_prompt + f",{lora_prompts[-1]}"
        processed = process_images(process_args2)
        gen_results = processed.images[0]

        progress.task_progress = 65
        progress.calc_eta_relative(upload_files_eta_secs)
        yield progress

        if second_lora_train_dataset:
            selected_face = self.select_high_quality_face(second_lora_train_dataset)
            # face_swap cv2
            swap_results = self.face_swap_fn(True, [gen_results], selected_face)
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

        progress.task_progress = 90
        progress.calc_eta_relative(upload_files_eta_secs)
        yield progress

        processed.images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in
                            final_gen_results_final]
        progress.task_progress = 95
        progress.status = TaskStatus.Uploading
        progress.calc_eta_relative(upload_files_eta_secs)
        yield progress

        images = save_processed_images(processed,
                                       process_args.outpath_samples,
                                       process_args.outpath_grids,
                                       process_args.outpath_scripts,
                                       t.id,
                                       inspect=False,
                                       forbidden_review=True)

        progress = TaskProgress.new_finish(t, images)
        yield progress







