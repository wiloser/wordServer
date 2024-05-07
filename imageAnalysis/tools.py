import cv2
import numpy as np


def rgb_to_rgba_with_transparent_background(image):
    # 将OpenCV的BGR格式转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将RGB转换为RGBA
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # 找到所有白色背景的像素
    # 这里我们假定白色为纯白，可以调整阈值以适应不完全是纯白的情况
    white = np.all(image == [255, 255, 255], axis=-1)

    # 将这些白色像素的透明度设置为0，从而使其变透明
    image_rgba[white, 3] = 0

    return image_rgba
