import math
import cv2
import numpy as np
import time
import os
from PIL import Image
import sys


# -----------------------------手写笔画坐标采集--------------------------------

class Pen:
    def __init__(self):
        self.save_txt_dir = "./bihua_data/"
        self.index = 0
        self.object_name = ["王"]
        self.object_type = ".txt"
        self.mode = "w"
        self.width = 256
        self.height = 256
        self.channles = 1
        self.step = 0
        self.flag = True
        while os.path.isfile(self.save_txt_dir + self.object_name[self.index] + self.object_type):
            print("[%s]字已存在，请删除后再修改" % self.object_name[self.index])
            self.index += 1
            if self.index >= len(self.object_name):
                self.flag = False
                break
        if self.flag:
            self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")
            add_point = '{},{}\n'.format(-1, -1)
            self.f.write(add_point)
        else:
            print("全部字体已存在，请进行删除后修改")

    def write(self, event, x, y, flags, parm):
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img, (x, y), 1, (0), 5, cv2.LINE_AA, 0)
            self.step += 1
            if self.step == 1:
                add_point = '{},{}\n'.format(x, y)
                self.f.write(add_point)
                self.step = 0
        if event == cv2.EVENT_LBUTTONUP:
            add_point = '{},{}\n'.format(-1, -1)
            self.f.write(add_point)
            print('下一笔')

    def init_write(self, img_init):
        for i in range(self.channles):
            img_init[:, :, i] = 255
        # 绘制米字格
        color = (200,)  # 浅红色
        thickness = 1
        cv2.line(img_init, (0, 0), (self.width, self.height), color, thickness)
        cv2.line(img_init, (self.width, 0), (0, self.height), color, thickness)
        cv2.line(img_init, (self.width // 2, 0), (self.width // 2, self.height), color, thickness)
        cv2.line(img_init, (0, self.height // 2), (self.width, self.height // 2), color, thickness)

    def next_write(self):
        self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")


pen_main = Pen()
if pen_main.flag:
    img = np.zeros((pen_main.width, pen_main.height, pen_main.channles), np.uint8)
    pen_main.init_write(img)
    print("第%s个字[%s]" % (pen_main.index + 1, pen_main.object_name[pen_main.index]))
    cv2.namedWindow("block")
    cv2.setMouseCallback("block", pen_main.write)
    while True:
        cv2.imshow("block", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            pen_main.f.close()
            break
        if k == ord('c'):
            pen_main.init_write(img)
            pen_main.f.close()
        if k == ord('n'):
            pen_main.f.close()
            pen_main.init_write(img)
            if pen_main.index < len(pen_main.object_name) - 1:
                pen_main.index += 1
                pen_main.next_write()
                print("下一个字[%s]" % pen_main.object_name[pen_main.index])
            else:
                print("没字了")
                break
    cv2.destroyAllWindows()


# -----------------------------对比评价--------------------------------
import math

def read_coords_from_txt(file_path):
    """
    从TXT文件读取坐标并组织成笔画列表
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords = []
    stroke = []
    for line in lines:
        x, y = map(int, line.strip().split(','))
        if (x, y) == (-1, -1):
            if stroke:
                coords.append(stroke)
                stroke = []
        else:
            stroke.append((x, y))

    if stroke:  # 如果最后一个笔画没有以-1, -1结尾
        coords.append(stroke)

    return coords

def calculate_midpoint(line):
    """
    计算直线的中点
    """
    x_coords = [point[0] for point in line]
    y_coords = [point[1] for point in line]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def calculate_length(line):
    """
    计算直线的长度
    """
    length = 0
    for i in range(1, len(line)):
        length += math.sqrt((line[i][0] - line[i-1][0])**2 + (line[i][1] - line[i-1][1])**2)
    return length

def calculate_angle(line):
    """
    计算直线的倾斜角度
    """
    dx = line[-1][0] - line[0][0]
    dy = line[-1][1] - line[0][1]
    return math.atan2(dy, dx) * 180 / math.pi

def check_intersection(line1, line2):
    """
    检查两条直线是否相交并计算夹角
    """
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def do_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        return False

    p1, q1 = line1[0], line1[-1]
    p2, q2 = line2[0], line2[-1]

    return do_intersect(p1, q1, p2, q2)

def compare_lines(line1, line2, stroke_index):
    """
    对比两条直线
    """
    print(f"笔画{stroke_index}对比:")

    # 计算中点
    midpoint1 = calculate_midpoint(line1)
    midpoint2 = calculate_midpoint(line2)

    # 判断中点是否在一条直线上
    on_same_line = math.isclose(midpoint1[1], midpoint2[1], abs_tol=1e-5)
    print(f"  - 临摹字笔画是否在模板字中点: {'是，笔画重心正确' if on_same_line else '否，笔画重心便宜'}")

    # 计算长度
    length1 = calculate_length(line1)
    length2 = calculate_length(line2)
    length_ratio = length2 / length1
    print(f"  - 长度比: {length_ratio:.2f}")

    if 0.95 <= length_ratio <= 1.05:
        print("  - 长度符合模板要求")
    elif length_ratio < 0.95:
        print("  - 临摹的笔画长度较短，应该拉长笔画")
    else:
        print("  - 临摹的笔画长度较长，应减小笔画长度")

    # 计算角度
    angle1 = calculate_angle(line1)
    angle2 = calculate_angle(line2)
    angle_diff = abs(angle1 - angle2)
    print(f"  - 角度差: {angle_diff:.2f}°")

    if angle_diff < 2:
        print("  - 角度临摹满分")
    elif 2 <= angle_diff < 5:
        print("  - 和模板略微有差异")
    else:
        print("  - 临摹角度极差，应该对汉字进行重新书写，要求更好")

    # 检查是否相交并计算夹角
    if check_intersection(line1, line2):
        print(f"  - 直线相交，夹角为: {angle_diff:.2f}°")
    else:
        if midpoint1[1] > midpoint2[1]:
            print("  - 临摹直线位于模板直线上方")
        else:
            print("  - 临摹直线位于模板直线下方")

    print("")  # 空行分隔每条笔画的结果

def main():
    original_coords_path = "王gif.txt"
    copied_coords_path = "王.txt"

    original_lines = read_coords_from_txt(original_coords_path)
    copied_lines = read_coords_from_txt(copied_coords_path)

    for index, (original_line, copied_line) in enumerate(zip(original_lines, copied_lines), start=1):
        compare_lines(original_line, copied_line, index)

if __name__ == "__main__":
    main()
