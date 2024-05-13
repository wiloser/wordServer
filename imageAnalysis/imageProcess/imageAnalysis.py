import cv2
import numpy as np
from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.losses import cosine_similarity
from keras.preprocessing import image


# 图片预处理
def ImagePreprocessing(imageData):
    imageData = np.array(imageData)
    gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
    # 通过阈值处理将汉字背景变为白色
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # 大于阈值时置 0(黑)，否则置 255（白）
    # 反转二值化图像，将汉字变为黑色，背景变为白色
    thresh = cv2.bitwise_not(thresh)
    # 执行形态学操作去除噪点
    kernel = np.ones((5, 5), np.uint8)
    imageData_result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return imageData_result


# 骨架提取
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
# todo:高
def SkeletonSimilarityEvaluation(imageData1, imageData2):
    # 提取特征向量,利用余弦进行相似度计算
    vector1 = np.array(extract_features(imageData1)).reshape(1, 25088)
    vector2 = np.array(extract_features(imageData2)).reshape(1, 25088)
    normalized_distance = cosine_similarity(vector1, vector2)
    # 转换 Tensor 为 numpy 数组并访问第一个元素
    value = normalized_distance.numpy()[0]
    print(value)
    return value.item()


# 笔画提取及相似度评价
def StrokeExtractionAndSimilarityEvaluation(imageData, reference_image):
    # 计算参考模块的Hu矩
    reference_hu_moments = calculate_hu_moments(reference_image)

    # 定义九宫格的位置和权重
    grid_positions = [(0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2)]

    grid_weights = [0.1, 0.1, 0.1,
                    0.1, 0.5, 0.1,
                    0.1, 0.1, 0.1]

    # 初始化存储每个模块Hu矩的列表
    hu_moments_list = []

    # 遍历九宫格的位置
    for position in grid_positions:
        # 提取当前模块的图像
        grid_image = extract_grid_image(imageData, position)
        # 计算当前模块的Hu矩
        hu_moments = calculate_hu_moments(grid_image)
        # 将Hu矩添加到列表中
        hu_moments_list.append(hu_moments)

    # 计算皮尔逊相关系数
    correlation_coefficients = []
    for hu_moments in hu_moments_list:
        correlation_coefficient = np.corrcoef(reference_hu_moments, hu_moments)[0, 1]
        correlation_coefficients.append(correlation_coefficient)

    # 加权求和
    weighted_sum = np.dot(correlation_coefficients, grid_weights)
    return weighted_sum


# 章法布局相似度及评价
def LayoutSimilarityEvaluation(imageData, reference_image):
    # 转换为灰度图像
    img1 = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # 将图像转换为二值图像
    _, img1_binary = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img2_binary = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算图像中的轮廓
    contours1, _ = cv2.findContours(img1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个字的自宽、自高
    box_widths1 = []
    box_heights1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        box_widths1.append(w)
        box_heights1.append(h)

    box_widths2 = []
    box_heights2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        box_widths2.append(w)
        box_heights2.append(h)

    # 计算每个字的布局特征
    layout_feature1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x
        right_distance = 399 - (x + w)
        top_distance = y
        bottom_distance = 399 - (y + h)
        layout_feature1.append([left_distance, right_distance, top_distance, bottom_distance])

    layout_feature2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x
        right_distance = 399 - (x + w)
        top_distance = y
        bottom_distance = 399 - (y + h)
        layout_feature2.append([left_distance, right_distance, top_distance, bottom_distance])

    # 计算布局相似度
    layout_similarities = []
    for i in range(len(layout_feature1)):
        for j in range(len(layout_feature2)):
            distance = np.sqrt(np.sum(np.square(np.array(layout_feature1[i]) - np.array(layout_feature2[j]))))
            similarity = 1 / (1 + distance)
            layout_similarities.append(similarity)

    return np.mean(layout_similarities)


# todo
def ImageAnalysis(imageData):
    # 进行图片分析
    return {
        # 识别为的文字
        'word': '王',
        # 得分
        'score': 9.0,
    }


# -----------------------------骨架提取--------------------------------

def VThin(image, array):
    h, w, _ = image.shape  # 图像高度和宽度
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                # 边界条件处理，确保不会越界
                M = (image[i, j - 1] + image[i, j] + image[i, j + 1]) if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    # 填充数组a，检查周围的像素
                    for k in range(3):
                        for l in range(3):
                            ii, jj = i - 1 + k, j - 1 + l
                            if 0 <= ii < h and 0 <= jj < w and image[ii, jj] == 255:
                                a[k * 3 + l] = 1
                    # 计算sum用于查表决定新的像素值
                    sum_val = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum_val] * 255
                    if array[sum_val] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h, w, _ = image.shape  # 图像高度和宽度
    NEXT = 1
    for j in range(h):  # 遍历所有行
        for i in range(w):  # 遍历所有列
            if NEXT == 0:
                NEXT = 1
            else:
                M = (image[j, i - 1] + image[j, i] + image[j, i + 1]) if 0 < i < w - 1 else 1
                if image[j, i] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            ii, jj = j - 1 + k, i - 1 + l
                            if 0 <= ii < h and 0 <= jj < w and image[ii, jj] == 255:
                                a[k * 3 + l] = 1
                    sum_val = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[j, i] = array[sum_val] * 255
                    if array[sum_val] == 1:
                        NEXT = 0
    return image


def Xihua(image, array, num=20):
    h = image.shape[0]
    w = image.shape[1]
    iXihua = np.zeros((h, w, 1), dtype=np.uint8)
    np.copyto(iXihua, image)
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua


def Two(image):
    h = image.shape[0]
    w = image.shape[1]

    iTwo = np.zeros((h, w, 1), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            iTwo[i, j] = 0 if image[i, j] < 200 else 255
    return iTwo


# -----------------------------相似度评价--------------------------------

# 加载图片并预处理
def load_and_process_image(imageData):
    img_array = image.img_to_array(imageData)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor


# 提取特征
def extract_features(imageData):
    model = VGG16(weights='imagenet', include_top=False)
    img_tensor = load_and_process_image(imageData)
    features = model.predict(img_tensor)
    # 将特征扁平化为一维数组
    flattened_features = features.flatten()
    return flattened_features


# -----------------------------笔画提取--------------------------------
# 定义函数计算图像的Hu矩
def calculate_hu_moments(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 计算Hu矩
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments)
    # 归一化处理
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + np.finfo(float).eps)
    return hu_moments.flatten()


def extract_grid_image(image, position):
    # 假设九宫格每个模块的大小是原图像的三分之一
    height, width = image.shape[:2]
    grid_height = height // 3
    grid_width = width // 3

    # 根据位置提取模块图像
    x, y = position
    start_x = x * grid_width
    start_y = y * grid_height
    end_x = start_x + grid_width
    end_y = start_y + grid_height

    # 确保提取范围在图像尺寸之内
    end_x = min(end_x, width)
    end_y = min(end_y, height)

    # 提取模块图像
    grid_image = image[start_y:end_y, start_x:end_x]
    return grid_image
