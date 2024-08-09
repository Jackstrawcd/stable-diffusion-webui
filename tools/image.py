#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 4:57 PM
# @Author  : wangdongming
# @Site    : 
# @File    : image.py
# @Software: Hifive
import os
import base64
import shutil
import numpy as np
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo

Image.MAX_IMAGE_PIXELS = 933120000


def encode_pil_to_base64(image, quality=50):
    with BytesIO() as output_bytes:
        use_metadata = False
        metadata = PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None),
                   quality=quality)
        bytes_data = output_bytes.getvalue()

        return 'data:image/png;base64,' + base64.b64encode(bytes_data).decode('ascii')


# compress_image 压缩图片函数，减轻网络压力
def compress_image(infile, outfile, kb=300, step=30, quality=70):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 输出路径。
    :param kb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件字节流
    """
    o_size = os.path.getsize(infile) / 1024
    # print(f'  > 原始大小：{o_size}')
    if o_size <= kb:
        # 大小满足要求
        shutil.copy(infile, outfile)

    pnginfo_data = PngInfo()
    im = Image.open(infile)
    if hasattr(im, "text"):
        for k, v in im.text.items():
            pnginfo_data.add_text(k, str(v))

    im = im.convert("RGB")  # 兼容处理png和jpg
    img_bytes = None

    while o_size > kb:
        out = BytesIO()
        im.save(out, format="JPEG", quality=quality, pnginfo=pnginfo_data)
        if quality - step < 0:
            break
        img_bytes = out.getvalue()
        o_size = len(img_bytes) / 1024
        out.close()  # 销毁对象
        quality -= step  # 质量递减
    if img_bytes:
        with open(outfile, "wb+") as f:
            f.write(img_bytes)
    else:
        shutil.copy(infile, outfile)


def thumbnail(infile, outfile, scale=0.4, w=0, h=0, quality=70):
    img = Image.open(infile)
    if w == 0 or h == 0:
        w, h = img.size
        w, h = round(w * scale), round(h * scale)

    img.thumbnail((w, h))
    img.save(outfile, optimize=True, quality=quality)
    img.close()


def plt_show(img, title=None):
    import matplotlib.pyplot as plt
    plt.title(title or 'undefined')
    plt.imshow(img)
    plt.show()


# 黑白照片（灰度图）识别
def is_gray_image(image_path, threshold=15):
    """
    入参：
    image_path：PIL读入的图像路径
    threshold：判断阈值，图片3个通道间差的方差均值小于阈值则判断为灰度图。
    阈值设置的越小，容忍出现彩色面积越小；设置的越大，那么就可以容忍出现一定面积的彩色，例如微博截图。
    如果阈值设置的过小，某些灰度图片会被漏检，这是因为某些黑白照片存在偏色，例如发黄的黑白老照片、
    噪声干扰导致灰度图不同通道间值出现偏差（理论上真正的灰度图是RGB三个通道的值完全相等或者只有一个通道，
    然而实际上各通道间像素值略微有偏差看起来仍是灰度图）
    出参：
    bool值
    """
    img = Image.open(image_path)
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False
