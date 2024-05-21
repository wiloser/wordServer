#encoding=utf-8
import cv2
import numpy as np
import time
import os.path

# 初始化类 Pen 的对象属性。
# 遍历 object_name 列表检查对应的文件是否存在，存在则提示用户删除后再修改。
# 打开当前汉字对应的文件以写入模式保存笔画数据，如果所有汉字文件都存在则提示并设置 flag 为 False。
class Pen:
    def __init__(self):
        self.save_txt_dir = "./bihua_data/"
        self.index = 0
        self.object_name = ["王", "永"]
        self.object_type = ".txt"
        self.mode = "w"
        self.width = 256
        self.height = 256
        self.channles = 1
        self.step = 0
        self.flag = True
        self.img = None
        self.f = None
        self.check_existing_files()
        if self.flag:
            self.init_file()
        else:
            print("全部字体已存在，请进行删除后修改")

    def check_existing_files(self):
        while os.path.isfile(self.save_txt_dir + self.object_name[self.index] + self.object_type):
            print("[%s]字已存在，请删除后再修改" % self.object_name[self.index])
            self.index += 1
            if self.index >= len(self.object_name):
                self.flag = False
                break

    def init_file(self):
        self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")
        add_point = '{},{}\n'.format(-1, -1)
        self.f.write(add_point)

    # wirte方法是鼠标事件的回调函数
    # 当鼠标移动且左键按下时，在图像上绘制一个点，并记录点的坐标到文件中
    # 当左键松开时，记录一个(-1, -1)，表示一笔结束
    def wirte(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.img, (x, y), 1, (0), 5, cv2.LINE_AA, 0)
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
    def init_write(self):
        self.img = np.zeros((self.width, self.height, self.channles), np.uint8)
        self.img.fill(255)

    #打开下一个汉字对应的文件以写入模式保存笔画数据
    def next_write(self):
        self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")


# 创建 Pen 类的实例 pen_main 并初始化。
# 如果 flag 为 True，则初始化图像并设置窗口和鼠标回调函数。
# 在循环中展示图像，监听按键操作：
# q 键：关闭当前文件并退出程序。
# c 键：清空当前画布并关闭当前文件。
# n 键：关闭当前文件，重置画布，处理下一个字。
# 当所有字都处理完毕时，退出程序并销毁窗口
    def run(self):
        if self.flag:
            self.init_write()
            print("第%s个字[%s]" % (self.index + 1, self.object_name[self.index]))
            cv2.namedWindow("block")
            cv2.setMouseCallback("block", self.wirte)
            while True:
                cv2.imshow("block", self.img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    self.f.close()
                    break
                if k == ord('c'):
                    self.init_write()
                    self.f.close()
                if k == ord('n'):
                    self.f.close()
                    self.init_write()
                    if self.index < len(self.object_name) - 1:
                        self.index += 1
                        self.next_write()
                        print("下一个字[%s]" % self.object_name[self.index])
                    else:
                        print("没字了")
                        break
            cv2.destroyAllWindows()

if __name__ == "__main__":
   pen_main = Pen()
   pen_main.run()