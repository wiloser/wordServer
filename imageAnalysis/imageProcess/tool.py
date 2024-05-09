import base64

import cv2
import numpy as np
from PIL import Image
import io


# 将base64编码的图片转换为cv2图片格式
def Base64ToImage(base64Data):
    # 从base64中提取图片数据
    imageData = base64Data.split(',', 1)[0]
    # 将图片数据转换为bytes类型
    imageData = base64.b64decode(imageData)
    # 将图片数据转换为cv2的Image类型
    imageData = Image.open(io.BytesIO(imageData))
    return imageData


# 将PIL Image转换为OpenCV图像，并转换为灰度
def PILImageToCVImage(imageData):
    # 将PIL Image转换为OpenCV图像
    imageData = cv2.cvtColor(np.array(imageData), cv2.COLOR_RGB2BGR)
    return imageData


# 将cv2图片格式转换为base64编码
def ImageToBase64(imageData):
    _imageData = rgb_to_rgba_with_transparent_background(imageData)
    # 将分析结果编码为PNG格式的字节流
    success, encoded_image = cv2.imencode('.png', _imageData)
    if not success:
        raise ValueError("Image encoding failed.")
    # 将字节流转换为base64编码的字符串
    imageData_result_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    print('result:', type(imageData_result_base64), imageData_result_base64, )
    # 序列化返回结果
    return imageData_result_base64


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
