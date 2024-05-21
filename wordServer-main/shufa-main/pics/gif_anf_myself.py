#encoding=utf-8
import cv2
import numpy as np
import os.path

# 初始化类 Pen 的对象属性。
# 遍历 object_name 列表检查对应的文件是否存在，存在则提示用户删除后再修改。
# 打开当前汉字对应的文件以写入模式保存笔画数据，如果所有汉字文件都存在则提示并设置 flag 为 False。
#手写汉字笔画提取
class Pen:
    def __init__(self):
        self.save_txt_dir = "./bihua_data/"
        self.index=0
        self.object_name = ["了", "永"]
        self.object_type = ".txt"
        self.mode="w"
        self.width = 256
        self.height = 256
        self.channles = 1
        self.step=0
        self.flag=True
        while(os.path.isfile(self.save_txt_dir+self.object_name[self.index]+self.object_type)):
            print("[%s]字已存在，请删除后再修改"%self.object_name[self.index])
            self.index+=1
            # print(self.index)
            if (self.index >= len(self.object_name)):
                self.flag=False
                break
        if(self.flag):
            self.f = open(self.save_txt_dir+self.object_name[self.index] + self.object_type, "w")
            add_point = '{},{}\n'.format(-1, -1)
            self.f.write(add_point)
        else:
            print("全部字体已存在，请进行删除后修改")

    # wirte方法是鼠标事件的回调函数
    # 当鼠标移动且左键按下时，在图像上绘制一个点，并记录点的坐标到文件中
    # 当左键松开时，记录一个(-1, -1)，表示一笔结束
    def wirte(self,event,x,y,flags,parm):
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img, (x, y), 1, (0), 5,cv2.LINE_AA,0)
            # cv2.circle(img, center=(x, y), radius=3,
            #            color=(0), thickness=-1)
            self.step += 1
            if self.step == 1:
                add_point = '{},{}\n'.format(x, y)
                self.f.write(add_point)
                self.step = 0
        if event == cv2.EVENT_LBUTTONUP:
            add_point = '{},{}\n'.format(-1, -1)
            self.f.write(add_point)
            print('下一笔')

    #将图像初始化为白色背景
    def init_write(self,img_init):
        for i in range(self.channles):
            img_init[:, :, i] = 255

    #打开下一个汉字对应的文件以写入模式保存笔画数据
    def next_write(self):
        self.f = open(self.save_txt_dir+self.object_name[self.index] + self.object_type, "w")



# 创建 Pen 类的实例 pen_main 并初始化。
# 如果 flag 为 True，则初始化图像并设置窗口和鼠标回调函数。
# 在循环中展示图像，监听按键操作：
# q 键：关闭当前文件并退出程序。
# c 键：清空当前画布并关闭当前文件。
# n 键：关闭当前文件，重置画布，处理下一个字。
# 当所有字都处理完毕时，退出程序并销毁窗口
pen_main=Pen()
if pen_main.flag:
    img=np.zeros((pen_main.width,pen_main.height,pen_main.channles),np.uint8)
    pen_main.init_write(img)
    print("第%s个字[%s]"%(pen_main.index+1,pen_main.object_name[pen_main.index]))
    cv2.namedWindow("block")
    cv2.setMouseCallback("block",pen_main.wirte)
    while(1):
        cv2.imshow("block",img)
        k = cv2.waitKey(1)&0xFF
        if k == ord('q'):
            pen_main.f.close()
            break
        if k == ord('c'):
            pen_main.init_write(img)
            pen_main.f.close()
        if k==ord('n'):
            pen_main.f.close()
            pen_main.init_write(img)
            if(pen_main.index<len(pen_main.object_name)-1):
                pen_main.index+=1
                pen_main.next_write()
                print("下一个字[%s]"%pen_main.object_name[pen_main.index])
            else:
                print("没字了")

                break
    cv2.destroyAllWindows()



from PIL import Image
import os
import cv2
import numpy as np
import math
import sys
import imutils

"""
    将一张GIF动图分解到指定文件夹
    src_path：要分解的gif的路径
    dest_path：保存后的gif路径
"""


class Gain():
    def __init__(self):
        self.index = 0
        self.save_txt_dir = "./gif_data/"
        self.object_name = ["王","永"]
        self.object_type = ".txt"
        self.image_type = ".gif"
        self.save_dir = './pics'
        self.mode = "w"
        self.width = 256
        self.height = 256
        self.channles = 1
        self.frame_only = False
        self.start = False
        self.step = 0
        self.flag = True
        self.dir = "./image/"
        while (os.path.isfile(self.save_txt_dir + self.object_name[self.index] + self.object_type)):
            print("[%s]字已存在，请删除后再修改" % self.object_name[self.index])
            self.index += 1
            # print(self.index)
            if (self.index >= len(self.object_name)):
                self.flag = False
                break
        if (self.flag):
            self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")
            self.flag=False
        else:
            print("全部字体已存在，请进行删除后修改")
            sys.exit()


    #计算两个点之间的欧几里得距离
    def cal_distance(self,p1, p2):
        return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

    #返回 p1 和 p2 之间等间距的中间点列表
    def intermediates(self,p1, p2, nb_points=8):
        """"Return a list of nb_points equally spaced points
        between p1 and p2"""
        # If we have 8 intermediate points, we have 8+1=9 spaces
        # between p1 and p2
        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

        return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
                for i in range(1, nb_points + 1)]

    #从轮廓列表中选择面积大于 MIN_AREA 的轮廓，并返回其中心点和大小
    def choose_contours(self, contours, MIN_AREA=20):
        point_list = []
        contour_list = []
        for item in contours:
            # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
            rect = cv2.boundingRect(item)
            width = rect[2]
            height = rect[3]
            x = (rect[0]*2+width)//2
            y = (rect[1]*2+height)//2
            # x=rect[0]
            # y=rect[1]
            # print(width / height)
            # print(width * height)
            # 计算方块的大小比例,符合的装到集合
            # print(width*height)
            if (width * height > MIN_AREA):
                point_list.append((x, y, width, height))
                # print(width * height)
                # print(width / height)
                contour_list.append(item)
        return point_list, contour_list

    #将点的坐标写入文件
    def txt_add_point(self, px, py):
        add_point = '{},{}\n'.format(int(px), int(py))
        self.f.write(add_point)


#将 GIF 动图分解为帧并处理每一帧。
#提取每帧的轮廓并计算轮廓中心点。
#将笔画信息（坐标）写入文件
    def gifSplit(self, src_path, dest_path, suffix="png"):
        frameNum = 0
        img = Image.open(src_path)
        point_x = -1
        point_y = -1
        pre_point_x,pre_point_y=-1,-1
        for i in range(img.n_frames):
            img.seek(i)
            # seek() 方法用于移动文件读取指针到指定位置。
            new = Image.new("RGB", img.size)
            # new=new.resize((width, height), Image.ANTIALIAS)
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
                #        img = cv2.imread("E:/chinese_ocr-master/4.png")
                #        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, threshold_frame = cv2.threshold(currentframe_open, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # ret, threshold_frame = cv2.threshold(currentframe_abs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)
                contours = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[1] if imutils.is_cv3() else contours[0]
                font_point_list, _ = self.choose_contours(contours)

                if (not font_point_list) and (self.flag):
                    if (self.frame_only):
                        # self.txt_add_point(point_x, point_y)
                        for point_and_point_x, point_and_point_y in self.intermediates([point_x - 10, point_y + 10],
                                                                                  [point_x, point_y], 12):
                            point_x, point_y = point_and_point_x, point_and_point_y
                            self.txt_add_point(point_x, point_y)
                        # point_x, point_y = point_x+10, point_y+10
                        # self.txt_add_point(point_x, point_y)
                        self.frame_only = False
                    pre_point_x,pre_point_y=point_x,point_y
                    point_x, point_y = -1, -1
                    self.txt_add_point(point_x, point_y)
                    self.flag = False
                elif (font_point_list):
                    for (x, y, width, height) in font_point_list:
                        if (point_x == -1) and (point_y == -1):
                            if pre_point_x!=-1 and pre_point_y!=-1 and self.cal_distance([pre_point_x,pre_point_y],[x,y])<=15:
                                for point_and_point_x, point_and_point_y in self.intermediates([pre_point_x,pre_point_y], [x, y]):
                                    self.txt_add_point(point_and_point_x, point_and_point_y)
                                    pre_point_x,pre_point_y=-1,-1
                            point_x, point_y = x, y
                            self.frame_only = True
                        else:
                            # print("距离",self.cal_distance([point_x,point_y],[x,y]))
                            if (self.cal_distance([point_x, point_y], [x, y]) >= 40):
                                point_x, point_y = x, y
                                self.txt_add_point(point_x, point_y)
                            # elif (self.cal_distance([point_x, point_y], [x, y]) <= 5):
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
                # cv2.imshow(str(i),new)

            cv2.waitKey(10)
        self.f.close()

    #打开下一个汉字对应的文件以写入模式保存笔画数据
    def next_write(self):
        self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")
        self.flag=False

    #调用 gifSplit 方法处理 GIF 图像并保存笔画数据
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
# gain.gifSplit('image/匧.gif', r'./pics')
gain.main()