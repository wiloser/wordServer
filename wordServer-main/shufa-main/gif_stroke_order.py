from PIL import Image
import os
import cv2
import numpy as np
import math
import sys
import imutils



#笔画判别
class StrokeComparer:
    def __init__(self, gif_strokes_file, handwritten_strokes_file, tolerance=10):
        self.gif_strokes = self.load_strokes(gif_strokes_file)
        self.handwritten_strokes = self.load_strokes(handwritten_strokes_file)
        self.tolerance = tolerance

    def load_strokes(self, file_path):
        strokes = []
        current_stroke = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                if x == -1 and y == -1:
                    strokes.append(current_stroke)
                    current_stroke = []
                else:
                    current_stroke.append((x, y))

        return strokes

    def compare_strokes(self):
        if len(self.gif_strokes) != len(self.handwritten_strokes):
            print("笔画数量不匹配")
            return False

        for gif_stroke, handwritten_stroke in zip(self.gif_strokes, self.handwritten_strokes):
            if not self.compare_single_stroke(gif_stroke, handwritten_stroke):
                print("笔画顺序或位置不符合")
                return False

        print("笔画顺序和位置符合")
        return True

    def compare_single_stroke(self, gif_stroke, handwritten_stroke):
        if len(gif_stroke) != len(handwritten_stroke):
            return False

        for (gx, gy), (hx, hy) in zip(gif_stroke, handwritten_stroke):
            if abs(gx - hx) > self.tolerance or abs(gy - hy) > self.tolerance:
                return False

        return True

gif_strokes_file = 'gif_data/王.txt'
handwritten_strokes_file = 'bihua_data/王.txt'

comparer = StrokeComparer(gif_strokes_file, handwritten_strokes_file)

if comparer.compare_strokes():
    print("手写笔画顺序符合标准")
else:
    print("手写笔画顺序不符合标准")
