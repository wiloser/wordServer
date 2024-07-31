from PIL import Image
import os
import cv2
import numpy as np
import math
import sys
import imutils

class Gain():
    def __init__(self):
        self.index = 0
        self.save_txt_dir = "./gif_data/"
        self.object_name = ["王","两","辉","中"]
        self.object_type = ".txt"
        self.image_type = ".gif"
        self.save_dir = './pics'
        self.mode = "w"
        self.width = 256
        self.height = 256
        self.channels = 1
        self.frame_only = False
        self.start = False
        self.step = 0
        self.flag = True
        self.dir = "./image/"
        while (os.path.isfile(self.save_txt_dir + self.object_name[self.index] + self.object_type)):
            print("[%s]字已存在，请删除后再修改" % self.object_name[self.index])
            self.index += 1
            if (self.index >= len(self.object_name)):
                self.flag = False
                break
        if (self.flag):
            self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")
            self.flag = False
        else:
            print("全部字体已存在，请进行删除后修改")
            sys.exit()

    def cal_distance(self, p1, p2):
        return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

    def intermediates(self, p1, p2, nb_points=8):
        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)
        return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing] for i in range(1, nb_points + 1)]

    def choose_contours(self, contours, MIN_AREA=20):
        point_list = []
        contour_list = []
        for item in contours:
            rect = cv2.boundingRect(item)
            width = rect[2]
            height = rect[3]
            x = (rect[0] * 2 + width) // 2
            y = (rect[1] * 2 + height) // 2
            if (width * height > MIN_AREA):
                point_list.append((x, y, width, height))
                contour_list.append(item)
        return point_list, contour_list

    def txt_add_point(self, px, py):
        add_point = '{},{}\n'.format(int(px), int(py))
        self.f.write(add_point)

    def gifSplit(self, src_path, dest_path, suffix="png"):
        frameNum = 0
        img = Image.open(src_path)
        point_x = -1
        point_y = -1
        pre_point_x, pre_point_y = -1, -1

        # 添加开头的分界点(-1, -1)
        self.txt_add_point(-1, -1)

        for i in range(img.n_frames):
            img.seek(i)
            new = Image.new("RGB", img.size)
            new.paste(img)
            new.save(os.path.join(self.save_dir, "%d.%s" % (i, suffix)))
            new = cv2.cvtColor(np.asarray(new), cv2.COLOR_RGBA2GRAY)
            new_img = cv2.resize(new, (self.width, self.height))
            frameNum += 1

            if (frameNum == 1):
                previousframe = new_img
            if (frameNum >= 2):
                currentframe = new_img
                currentframe_abs = cv2.absdiff(currentframe, previousframe)
                currentframe_median = cv2.medianBlur(currentframe_abs, 3)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                currentframe_open = cv2.morphologyEx(currentframe_median, cv2.MORPH_OPEN, kernel1)
                currentframe_close = cv2.morphologyEx(currentframe_open, cv2.MORPH_CLOSE, kernel)
                currentframe_close = cv2.morphologyEx(currentframe_close, cv2.MORPH_CLOSE, kernel)
                currentframe_open = cv2.morphologyEx(currentframe_close, cv2.MORPH_OPEN, kernel1)
                ret, threshold_frame = cv2.threshold(currentframe_open, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[1] if imutils.is_cv3() else contours[0]
                font_point_list, _ = self.choose_contours(contours)

                if (not font_point_list) and (self.flag):
                    if (self.frame_only):
                        for point_and_point_x, point_and_point_y in self.intermediates([point_x - 10, point_y + 10], [point_x, point_y], 12):
                            point_x, point_y = point_and_point_x, point_and_point_y
                            self.txt_add_point(point_x, point_y)
                        self.frame_only = False
                    pre_point_x, pre_point_y = point_x, point_y
                    point_x, point_y = -1, -1
                    self.txt_add_point(point_x, point_y)
                    self.flag = False
                elif (font_point_list):
                    for (x, y, width, height) in font_point_list:
                        if (point_x == -1) and (point_y == -1):
                            if pre_point_x != -1 and pre_point_y != -1 and self.cal_distance([pre_point_x, pre_point_y], [x, y]) <= 15:
                                for point_and_point_x, point_and_point_y in self.intermediates([pre_point_x, pre_point_y], [x, y]):
                                    self.txt_add_point(point_and_point_x, point_and_point_y)
                                pre_point_x, pre_point_y = -1, -1
                            point_x, point_y = x, y
                            self.frame_only = True
                        else:
                            if (self.cal_distance([point_x, point_y], [x, y]) >= 40):
                                point_x, point_y = x, y
                                self.txt_add_point(point_x, point_y)
                            else:
                                self.txt_add_point(point_x, point_y)
                                for point_and_point_x, point_and_point_y in self.intermediates([point_x, point_y], [x, y]):
                                    point_x, point_y = point_and_point_x, point_and_point_y
                                    self.txt_add_point(point_x, point_y)
                                point_x, point_y = x, y
                                self.txt_add_point(point_x, point_y)
                            self.frame_only = False
                    self.flag = True
                cv2.imshow("Frame", threshold_frame)

                previousframe = currentframe
            cv2.waitKey(10)
        self.f.close()

    def next_write(self):
        self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")
        self.flag = False

    def collect(self):
        self.gifSplit((self.dir + self.object_name[self.index] + self.image_type), (self.image_type))
        self.f.close()

    def main(self):
        print("第%s个字[%s]" % (self.index + 1, self.object_name[self.index]))
        print("开始采集")
        self.collect()
        print("采集结束")
        while (1):
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                self.f.close()
                break
            if (k == ord('c')) and (self.start):
                print("开始采集")
                self.collect()
                print("采集结束")
            if k == ord('n'):
                self.f.close()
                self.start = True
                if (self.index < len(self.object_name) - 1):
                    self.index += 1
                    self.next_write()
                    print("下一个字[%s]" % self.object_name[self.index])
                else:
                    print("没字了")
                    break

gain = Gain()
gain.main()
