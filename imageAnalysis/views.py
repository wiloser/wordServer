import cv2
from PIL import Image
from rest_framework.views import APIView
from imageAnalysis.imageProcess.imageAnalysis import ImagePreprocessing, ImageSkeletonExtraction, \
    SkeletonSimilarityEvaluation, StrokeExtractionAndSimilarityEvaluation, LayoutSimilarityEvaluation
from imageAnalysis.imageProcess.tool import Base64ToImage, ImageToBase64, PILImageToCVImage
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
        # 将base64编码的图片转换为转换为cv2的Image类型
        imageData = Base64ToImage(data)
        # 检查图像是否包含透明度（即是否为'RGBA'）
        if imageData.mode == 'RGBA':
            # 创建一个白色背景
            white_background = Image.new('RGBA', imageData.size, (255, 255, 255, 255))
            # 将原图像粘贴到白色背景上，使用透明度通道作为掩码
            white_background.paste(imageData, mask=imageData.split()[3])
            imageData = white_background.convert('RGB')  # 转换为不带透明度的RGB格式
        # 修改图片大小
        imageData = imageData.resize((224, 224), Image.LANCZOS)
        imageData = PILImageToCVImage(imageData)
        imageData_template = cv2.imread('images/Template_pictures/1.png')

        # todo
        # 骨架相似度评价
        result_1 = SkeletonSimilarityEvaluation(imageData, imageData)

        # 笔画提取及相似度评价
        result_2 = StrokeExtractionAndSimilarityEvaluation(imageData, imageData)

        # 章法布局相似度及评价
        result_3 = LayoutSimilarityEvaluation(imageData, imageData)

        # 扣除黑色部分
        imageData_template = ImagePreprocessing(imageData_template)
        # 将cv2图片格式转换为base64编码
        imageData_result_base64 = ImageToBase64(imageData_template)
        # 传输大小
        print('result:', type(imageData_result_base64), len(imageData_result_base64), )

        return R.ok(
            data={
                'imageData': imageData_result_base64,
                'result': [
                    ['骨架相似度', result_1],
                    ['笔画相似度', result_2],
                    ['章法布局相似度', result_3],
                ]
            },
            msg='操作成功',
        )


class GetImages(APIView):
    def get(self, request):
        return R.ok(
            data={
                'words': ['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '吴', '周', '徐', '孙', '马', '朱', '胡',
                          '林', '郭', '何', '高', '罗', '郑', '梁', '谢', '宋', '唐', '许', '邓', '冯', '韩', '曹',
                          '曾', '彭', '萧', '蔡', '潘', '田', '董', '袁', '于', '余', '叶', '蒋', '杜', '苏', '魏',
                          '程', '吕', '丁', '沈', '任', '姚', '卢', '傅', '钟', '姜', '崔', '谭', '廖', '范', '汪',
                          '陆', '金', '石', '戴', '贾', '韦', '夏', '邱', '方', '侯', '邹', '熊', '孟', '秦', '白',
                          '江', '阎', '薛', '尹', '段', '雷', '黎', '史', '龙', '陶', '贺', '顾', '毛', '郝', '龚',
                          '邵', '万', '钱', '严', '赖', '覃', '洪', '武', '莫', '孔', '汤', '向', '常', '温', '康',
                          '施', '文', '牛', '樊', '葛', '邢', '安', '齐', '易', '乔', '伍', '庞', '颜', '倪', '庄',
                          '聂', '章', '鲁', '岳', '翟', '殷', '詹', '申', '欧', ],
            },
            msg='操作成功',
        )
