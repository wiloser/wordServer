import os

import cv2
import numpy as np
from PIL import Image


class Gain:
    def __init__(self):
        self.index = 0  # 当前正在处理的汉字的索引
        self.save_txt_dir = "./bihua_data/"
        self.object_name = ["王"]
        self.correct_stroke_orders = {
            "王": ["笔画1", "笔画2", "笔画3", "笔画4"]
        }
        self.correct_stroke_coords = {
            "王": [(50, 50), (150, 50), (100, 100), (50, 150)]  # 示例坐标
        }
        self.object_type = ".txt"
        self.image_type = ".gif"
        self.save_dir = './pics'
        self.width = 256
        self.height = 256
        self.frame_accumulator = []  # 累积整个字的笔画顺序
        self.start = False
        self.dir = "./image/"
        self.flag = True
        self.imitated_stroke_coords = []

    def determine_stroke_order(self, point_list):
        if not point_list:
            return []
        sorted_points = sorted(point_list, key=lambda p: (p[1], p[0]))
        stroke_order = [f"笔画{i + 1}" for i, _ in enumerate(sorted_points)]
        self.frame_accumulator.extend(stroke_order)
        return stroke_order

    def check_stroke_order(self):
        correct_order = self.correct_stroke_orders.get(self.object_name[self.index], [])
        if self.frame_accumulator == correct_order:
            print("笔画顺序正确，请继续书写")
            return True
        else:
            print("笔画顺序错误，请重新开始书写")
            return False

    def check_stroke_coords(self):
        correct_coords = self.correct_stroke_coords.get(self.object_name[self.index], [])
        if not correct_coords or not self.imitated_stroke_coords:
            return False

        # 判断临摹字的笔画顺序是否与被临摹字匹配
        if len(correct_coords) != len(self.imitated_stroke_coords):
            print("笔画数量不匹配，请重新开始书写")
            return False

        for correct, imitated in zip(correct_coords, self.imitated_stroke_coords):
            if np.linalg.norm(np.array(correct) - np.array(imitated)) > 10:  # 阈值可以调整
                print("笔画位置错误，请重新开始书写")
                return False

        print("笔画顺序正确，请继续书写")
        return True

    def simulate_frame_processing(self, frame_img):
        thresh = cv2.adaptiveThreshold(frame_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(thresh, 50, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stroke_info = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                stroke_info.append((cx, cy))

        stroke_info = sorted(stroke_info, key=lambda x: x[1])
        return stroke_info

    def gifSplit(self, src_path, dest_path, suffix="png"):
        img = Image.open(src_path)

        for i in range(img.n_frames):
            img.seek(i)
            new = Image.new("RGB", img.size)
            new.paste(img)
            frame_path = os.path.join(self.save_dir, "%d.%s" % (i, suffix))
            new.save(frame_path)
            new = cv2.cvtColor(np.asarray(new), cv2.COLOR_RGB2GRAY)
            new_img = cv2.resize(new, (self.width, self.height))
            font_point_list = self.simulate_frame_processing(new_img)
            self.imitated_stroke_coords.append(font_point_list)
            stroke_order = self.determine_stroke_order(font_point_list)
            print(f"第 {i + 1} 帧的笔画顺序：{stroke_order}")

        if not self.check_stroke_order() or not self.check_stroke_coords():
            self.frame_accumulator = []
            self.imitated_stroke_coords = []
            return

    def main(self):
        self.gifSplit(self.dir + self.object_name[self.index] + self.image_type, self.image_type)


gain = Gain()
gain.main()
