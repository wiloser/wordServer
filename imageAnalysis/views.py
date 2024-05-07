from rest_framework.views import APIView
import base64
from wordServer.utils import R


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
        imageData = data.split(',', 1)[1]
        # 将图片数据转换为bytes类型
        imageData = base64.b64decode(imageData)
        # 进行图片分析

        return R.ok(data=data, msg='操作成功')