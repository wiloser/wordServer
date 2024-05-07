import cv2
import numpy as np
from rest_framework.views import APIView
import base64
from PIL import Image
from imageAnalysis.imageAnalysis import ImagePreprocessing, ImageSkeletonExtraction
from imageAnalysis.tools import rgb_to_rgba_with_transparent_background
from wordServer.utils import R
import io


# Create your views here.
class Analysis(APIView):
    def get(self, request):
        # 得到图片数据
        data = request.data['imageData']
        return R.ok(data, '操作成功')

    def post(self, request):
        print(request.data)
        # 得到图片数据
        data = request.data['imageData']
        # 从base64中提取图片数据
        imageData = data.split(',', 1)[0]
        # 将图片数据转换为bytes类型
        imageData = base64.b64decode(imageData)
        # 将图片数据转换为cv2的Image类型
        imageData = Image.open(io.BytesIO(imageData))
        imageData.save('images/image.png')
        # 检查图像是否包含透明度（即是否为'RGBA'）
        if imageData.mode == 'RGBA':
            # 创建一个白色背景
            white_background = Image.new('RGBA', imageData.size, (255, 255, 255, 255))
            # 将原图像粘贴到白色背景上，使用透明度通道作为掩码
            white_background.paste(imageData, mask=imageData.split()[3])
            imageData = white_background.convert('RGB')  # 转换为不带透明度的RGB格式
        # 存储到本地
        imageData = np.array(imageData)
        # 进行图片分析
        # todo
        imageAnalysis_result = ImagePreprocessing(imageData)
        # 去掉白色背景
        imageAnalysis_result = rgb_to_rgba_with_transparent_background(imageAnalysis_result)
        # 存储到本地
        cv2.imwrite('images/imageAnalysis_result.png', imageAnalysis_result)
        # 将分析结果编码为PNG格式的字节流
        success, encoded_image = cv2.imencode('.png', imageAnalysis_result)
        if not success:
            raise ValueError("Image encoding failed.")
        # 将字节流转换为base64编码的字符串
        imageData_result_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        print('result:', type(imageData_result_base64), imageData_result_base64, )
        # 序列化返回结果
        return R.ok(data=imageData_result_base64, msg='操作成功')
