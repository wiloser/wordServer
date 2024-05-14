import cv2
import numpy as np
from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.losses import cosine_similarity
from keras.preprocessing import image
from sklearn.metrics import euclidean_distances


# 图片预处理
def ImagePreprocessing(imageData,target_size=(224, 224)):
    imageData = np.array(imageData)
    gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
    # 通过阈值处理将汉字背景变为白色
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # 大于阈值时置 0(黑)，否则置 255（白）
    # 反转二值化图像，将汉字变为黑色，背景变为白色
    thresh = cv2.bitwise_not(thresh)
    # 执行形态学操作去除噪点
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # 对图片的尺寸进行设定
    imageData_resized = cv2.resize(opening, target_size, interpolation=cv2.INTER_AREA)
    return imageData_resized


# 骨架提取,调用VThin、HThin、Xihua、Two函数，对预处理后的图片进行汉字骨架细化
def ImageSkeletonExtraction(imageData):
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    # 进行骨架提取
    iTwo = Two(imageData)
    iThin = Xihua(iTwo, array)
    return iThin


# 骨架相似度评价
def SkeletonSimilarityEvaluation(imageData1, imageData2):
    # 提取特征向量,利用余弦进行相似度计算
    vector1 = np.array(extract_features(imageData1)).reshape(1, -1)
    vector2 = np.array(extract_features(imageData2)).reshape(1, -1)
    # 正规化向量
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    # 计算欧氏距离
    distance = euclidean_distances(vector1, vector2)
    # 转换距离为相似度
    similarity = 1 / (1 + distance)
    similarity *= 100  # 转换为百分比

    # 提取出单个数值
    similarity_value = similarity[0][0]  # 访问数组中的第一个元素
    similarity_value = round(similarity_value , 2)
    # 根据相似度值返回相应的描述
    if similarity_value < 20:
        description = '骨架极其不相似，与模板差异极大'
    elif similarity_value < 40:
        description = '骨架很不相似，需要大幅度调整'
    elif similarity_value < 60:
        description = '骨架不太相似，部分偏离模板'
    elif similarity_value < 80:
        description = '骨架位置基本相似，但仍有改进空间'
    else:
        description = '骨架位置非常相似，与模板高度一致'

    # 打印并返回相似度值和描述
    print('骨架相似度:', similarity_value)
    print(description)

    return similarity_value, description


# 笔画提取及相似度评价
def StrokeExtractionAndSimilarityEvaluation(imageData, reference_image):

    handwriting_hu_moments = calculate_hu_moments(imageData)
    template_hu_moments = calculate_hu_moments(reference_image)

    grid_positions = [(0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2)]

    grid_weights = [0.1, 0.1, 0.1,
                    0.1, 0.5, 0.1,
                    0.1, 0.1, 0.1]

    handwriting_hu_moments_list = []
    template_hu_moments_list = []

    for position in grid_positions:
        handwriting_grid_image = extract_grid_image(imageData, position)
        template_grid_image = extract_grid_image(reference_image, position)

        handwriting_hu_moments = calculate_hu_moments(handwriting_grid_image)
        template_hu_moments = calculate_hu_moments(template_grid_image)

        handwriting_hu_moments_list.append(handwriting_hu_moments)
        template_hu_moments_list.append(template_hu_moments)

    correlation_coefficients = []
    for handwriting_hu_moments, template_hu_moments in zip(handwriting_hu_moments_list, template_hu_moments_list):
        correlation_coefficient = np.corrcoef(handwriting_hu_moments, template_hu_moments)[0, 1]
        correlation_coefficients.append(correlation_coefficient)

    weighted_sum = np.dot(correlation_coefficients, grid_weights)
    weighted_sum = weighted_sum*76.924
    weighted_sum = round(weighted_sum , 2)

    # 根据相似度值返回相应的描述
    if weighted_sum < 20:
        description = '笔画极其不相似，与模板差异极大'
    elif weighted_sum < 40:
        description = '笔画很不相似，需要大幅度改进'
    elif weighted_sum < 60:
        description = '笔画不太相似，部分偏离模板'
    elif weighted_sum < 80:
        description = '笔画位置基本相似，但仍有改进空间'
    else:
        description = '笔画位置非常相似，与模板高度一致'

    print('笔画相似度', weighted_sum)
    print(description)
    return weighted_sum , description


# 章法布局相似度及评价
def LayoutSimilarityEvaluation(imageData, reference_image):
    # 将图像转换为二值图像
    _, img1_binary = cv2.threshold(imageData, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img2_binary = cv2.threshold(reference_image, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算图像中的轮廓
    contours1, _ = cv2.findContours(img1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个轮廓的布局特征
    layout_features1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x / imageData.shape[1]  # 归一化到[0, 1]
        right_distance = (imageData.shape[1] - (x + w)) / imageData.shape[1]
        top_distance = y / imageData.shape[0]
        bottom_distance = (imageData.shape[0] - (y + h)) / imageData.shape[0]
        layout_features1.append([left_distance, right_distance, top_distance, bottom_distance, w / imageData.shape[1],
                                 h / imageData.shape[0]])  # 添加字宽和字高

    layout_features2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x / reference_image.shape[1]
        right_distance = (reference_image.shape[1] - (x + w)) / reference_image.shape[1]
        top_distance = y / reference_image.shape[0]
        bottom_distance = (reference_image.shape[0] - (y + h)) / reference_image.shape[0]
        layout_features2.append(
            [left_distance, right_distance, top_distance, bottom_distance, w / reference_image.shape[1], h / reference_image.shape[0]])

        # 计算布局相似度
    layout_similarities = []
    for feature1 in layout_features1:
        max_similarity = 0
        for feature2 in layout_features2:
            distance = np.sqrt(np.sum(np.square(np.array(feature1[:-2]) - np.array(feature2[:-2]))))  # 忽略字宽和字高
            similarity = 1 / (1 + distance)  # 简化的相似度度量
            if similarity > max_similarity:
                max_similarity = similarity
        layout_similarities.append(max_similarity)
        # 计算平均相似度并乘以100转换为百分比
        average_similarity = np.mean(layout_similarities) * 100
        average_similarity = round(average_similarity, 2)

        # 根据相似度值返回相应的描述
        if average_similarity < 20:
            description = '布局极其混乱，与模板差异极大'
        elif average_similarity < 40:
            description = '布局很不规整，需要大幅度调整'
        elif average_similarity < 60:
            description = '布局不太规整，部分偏离模板'
        elif average_similarity < 80:
            description = '布局基本规整，但仍有改进空间'
        else:
            description = '布局非常规整，与模板高度一致'

        print('布局相似度', average_similarity)
        print(description)
        return average_similarity , description

def ImageAnalysis(imageData):
    # 进行图片分析
    return {
        # 识别为的文字
        'word': '王',
        # 得分
        'score': 9.0,
    }


# -----------------------------骨架提取--------------------------------

def VThin(imageData, array):
    h, w, _ = imageData.shape  # 图像高度和宽度
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                # 边界条件处理，确保不会越界
                M = (imageData[i, j - 1] + imageData[i, j] + imageData[i, j + 1]) if 0 < j < w - 1 else 1
                if imageData[i, j] == 0 and M != 0:
                    a = [0] * 9
                    # 填充数组a，检查周围的像素
                    for k in range(3):
                        for l in range(3):
                            ii, jj = i - 1 + k, j - 1 + l
                            if 0 <= ii < h and 0 <= jj < w and imageData[ii, jj] == 255:
                                a[k * 3 + l] = 1
                    # 计算sum用于查表决定新的像素值
                    sum_val = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    imageData[i, j] = array[sum_val] * 255
                    if array[sum_val] == 1:
                        NEXT = 0
    return image


def HThin(imageData, array):
    h, w, _ = imageData.shape  # 图像高度和宽度
    NEXT = 1
    for j in range(h):  # 遍历所有行
        for i in range(w):  # 遍历所有列
            if NEXT == 0:
                NEXT = 1
            else:
                M = (imageData[j, i - 1] + imageData[j, i] + imageData[j, i + 1]) if 0 < i < w - 1 else 1
                if imageData[j, i] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            ii, jj = j - 1 + k, i - 1 + l
                            if 0 <= ii < h and 0 <= jj < w and imageData[ii, jj] == 255:
                                a[k * 3 + l] = 1
                    sum_val = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    imageData[j, i] = array[sum_val] * 255
                    if array[sum_val] == 1:
                        NEXT = 0
    return image


def Xihua(imageData, array, num=20):
    h = imageData.shape[0]
    w = imageData.shape[1]
    iXihua = np.zeros((h, w, 1), dtype=np.uint8)
    np.copyto(iXihua, imageData)
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua


def Two(imageData):
    h = imageData.shape[0]
    w = imageData.shape[1]

    iTwo = np.zeros((h, w, 1), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            iTwo[i, j] = 0 if imageData[i, j] < 200 else 255
    return iTwo



# -----------------------------骨架相似度评价--------------------------------

# 加载图片并预处理
def load_and_process_image(imageData):
    # 初始化img_tensor为None，如果图像已经是彩色，则稍后会被覆盖
    img_tensor = None
    # 检查图像是否是灰度图像（形状为(height, width)）
    if len(imageData.shape) == 2:
        # 将灰度值复制到三个通道以创建伪彩色图像
        img_array_rgb = np.stack((imageData,) * 3, axis=-1)
        # 调整图像大小并添加batch维度
        img_tensor = np.expand_dims(img_array_rgb, axis=0)
    else:
        # 如果图像已经是彩色的，则直接调整大小并添加batch维度
        img_tensor = np.expand_dims(imageData, axis=0)
    assert img_tensor is not None, "img_tensor was not properly defined"

    return img_tensor



# 提取特征
def extract_features(imageData):
    model = VGG16(weights='imagenet', include_top=False)
    img_tensor = load_and_process_image(imageData)
    img_tensor = np.tile(img_tensor, (1, 1, 1, 3))
    img_tensor = img_tensor[:, :, :, :3]
    features = model.predict(img_tensor)
    return features


# -----------------------------笔画提取--------------------------------
# 定义函数计算图像的Hu矩
def calculate_hu_moments(imageData):
    ret, binary = cv2.threshold(imageData, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + np.finfo(float).eps)
    return hu_moments.flatten()


def extract_grid_image(imageData, position):
    height, width = imageData.shape[:2]
    grid_height = height // 3
    grid_width = width // 3
    x, y = position
    start_x = x * grid_width
    start_y = y * grid_height
    end_x = start_x + grid_width
    end_y = start_y + grid_height
    end_x = min(end_x, width)
    end_y = min(end_y, height)
    grid_image = imageData[start_y:end_y, start_x:end_x]
    return grid_image

# -----------------------------布局相似度--------------------------------
#先直接对原图进行Contour_extraction函数进行轮廓提取
def Contour_extraction(imageData):

    img = cv2.GaussianBlur(imageData, (3, 3), 3)
    th = cv2.adaptiveThreshold(img, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 11)
    th = cv2.bitwise_not(th)
    kernel = np.array([[0, 1, 1],
                           [0, 1, 0],
                           [1, 1, 0]], dtype='uint8')
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th


# -----------------------------高好晴check--------------------------------
#模板字和手写字读取
template_img_path  = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\1.png'
handwriting_img = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\4.png'

handwriting_img =  cv2.imread(handwriting_img)
template_img =  cv2.imread(template_img_path)

#图片预处理，使用image_preprocessing函数
image1 = ImagePreprocessing(handwriting_img)
image2 = ImagePreprocessing(template_img)

#预处理之后进行细化处理
xihuaimage1 = ImageSkeletonExtraction(image1)
xihuaimage2 = ImageSkeletonExtraction(image2)
# cv2.imshow('1',xihuaimage1)
# cv2.imshow('2',xihuaimage2)
# cv2.destroyAllWindows()

#骨架相似度评价
Skeleton_score = SkeletonSimilarityEvaluation(xihuaimage1, xihuaimage2)

#笔画相似度评价
weighted_score = StrokeExtractionAndSimilarityEvaluation(xihuaimage1, xihuaimage2)

#章法布局相似度及评价
img_contour1 = Contour_extraction(image1)
img_contour2 = Contour_extraction(image2)

#整体评价
skeleton_score, _ = SkeletonSimilarityEvaluation(xihuaimage1, xihuaimage2)
stroke_score, _ = StrokeExtractionAndSimilarityEvaluation(xihuaimage1, xihuaimage2)
layout_score, _ = LayoutSimilarityEvaluation(img_contour1, img_contour2)

# 计算总分
full_score = skeleton_score * 0.5 + stroke_score * 0.2 + layout_score * 0.3
print("总分：", full_score)