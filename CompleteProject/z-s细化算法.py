import cv2
import numpy as np
import matplotlib.pyplot as plt

# Zhang Suen thinning algorithm
def Zhang_Suen_thinning(img):
    # Get image shape
    H, W, C = img.shape

    # Prepare output image
    out = np.zeros((H, W), dtype=np.int32)
    out[img[..., 0] > 0] = 1

    # Inverse image (white background, black character)
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # Step 1 (raster scan)
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # Skip non-edge pixels
                if out[y, x] > 0:
                    continue

                # Condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1: f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1: f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1: f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1: f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1: f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1: f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1: f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1: f1 += 1

                if f1 != 1:
                    continue

                # Condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # Condition 4 and 5
                if (out[y - 1, x] + out[y, x + 1] + out[y + 1, x]) < 1: continue
                if (out[y, x + 1] + out[y + 1, x] + out[y, x - 1]) < 1: continue

                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # Step 2 (raster scan)
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                if out[y, x] > 0:
                    continue

                # Same conditions as step 1
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1: f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1: f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1: f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1: f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1: f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1: f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1: f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1: f1 += 1

                if f1 != 1:
                    continue

                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                if (out[y - 1, x] + out[y, x + 1] + out[y, x - 1]) < 1: continue
                if (out[y - 1, x] + out[y + 1, x] + out[y, x - 1]) < 1: continue

                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        if len(s1) < 1 and len(s2) < 1:
            break

    out = apply_template_removal(out, H, W)

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out

# Apply the removal template
def apply_template_removal(out, H, W):
    for y in range(1, H-1):
        for x in range(1, W-1):
            P = out[y-1:y+2, x-1:x+2].flatten()
            if (P[1] * P[7] == 1 and sum([P[3], P[4], P[5], P[8]]) == 0) or \
               (P[5] * P[7] == 1 and sum([P[1], P[2], P[3], P[6]]) == 0) or \
               (P[1] * P[3] == 1 and sum([P[2], P[5], P[6], P[7]]) == 0) or \
               (P[3] * P[5] == 1 and sum([P[1], P[4], P[7], P[8]]) == 0) or \
               (sum([P[2], P[4], P[6], P[8]]) == 0 and sum([P[1], P[3], P[5], P[7]]) == 3):
                out[y, x] = 1
    return out


# 定义模板
def apply_template(image, template):
    """根据给定模板进行处理，消除噪声或孔洞"""
    h, w = image.shape
    processed_image = image.copy()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 获取8邻域
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]
            # 与模板匹配
            if np.array_equal(neighborhood, template):
                processed_image[i, j] = 0  # 将符合条件的前景点更改为背景点
    return processed_image


# 凹凸点和孤立点模板
templates_bumps_isolated = [
    np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]]),

]

# 孔洞消除模板
templates_holes = [
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),

]

# 应用模板处理图像
def preprocess_image(image):
    # 处理凹凸点和孤立点
    for template in templates_bumps_isolated:
        image = apply_template(image, template)

    # 处理孔洞
    for template in templates_holes:
        image = apply_template(image, template)

    return image

# Read the image
image_path = r"E:\python files\Conda_env_files\pythonProject1\wordServer-main\imageAnalysis\imageProcess\4.png"
img = cv2.imread(image_path)
img =cv2.bitwise_not(img)

# Apply Zhang-Suen thinning with removal templates
out = Zhang_Suen_thinning(img)
out1 =preprocess_image(out)

# # Display the result
cv2.imshow("Thinned Image", out1)
cv2.waitKey(0)
cv2.destroyAllWindows()
