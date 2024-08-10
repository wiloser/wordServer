import numpy as np
import math

#########################  判断笔画个数和顺序        #######################
def read_coordinates_from_file(file_path):
    """从txt文件中读取坐标数据"""
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            # 移除空白字符
            line = line.strip()
            # 解析坐标
            if line:
                x, y = map(int, line.strip("()").split(","))
                coordinates.append((x, y))
    return coordinates


def normalize_coordinates(coordinates):
    """将坐标进行归一化处理"""
    coordinates = np.array(coordinates)
    # 去除分隔符 (-1, -1)
    valid_coords = coordinates[coordinates[:, 0] != -1]
    # 坐标大小归一化
    min_coords = valid_coords.min(axis=0)
    max_coords = valid_coords.max(axis=0)
    normalized_coords = (valid_coords - min_coords) / (max_coords - min_coords)

    # 重新插入分隔符 (-1, -1)
    result = []
    i = 0
    for coord in coordinates:
        if coord[0] == -1 and coord[1] == -1:
            result.append((-1, -1))
        else:
            result.append(tuple(normalized_coords[i]))
            i += 1
    return result


def split_strokes(coordinates):
    """根据 (-1, -1) 分割坐标，返回笔画列表"""
    strokes = []
    stroke = []
    for point in coordinates:
        if point == (-1, -1):
            if stroke:
                strokes.append(stroke)
                stroke = []
        else:
            stroke.append(point)
    if stroke:  # 添加最后一个笔画
        strokes.append(stroke)
    return strokes


def compare_strokes(strokes1, strokes2):
    """比较两个笔画序列的顺序是否一致，并输出具体哪个笔画不一致"""
    if len(strokes1) != len(strokes2):
        print(f"笔画数量不一致。手写字有 {len(strokes1)} 个笔画，GIF汉字有 {len(strokes2)} 个笔画。")
        return False

    for i, (stroke1, stroke2) in enumerate(zip(strokes1, strokes2), start=1):
        if not are_strokes_similar(stroke1, stroke2):
            print(f"第 {i} 个笔画不一致。")
            return False
        else:
            print(f"第 {i} 个笔画一致。")

    return True


def are_strokes_similar(stroke1, stroke2, tolerance=0.2):
    """判断两个笔画是否相似，通过起点和终点比较"""
    start1, end1 = stroke1[0], stroke1[-1]
    start2, end2 = stroke2[0], stroke2[-1]

    return (distance(start1, start2) < tolerance and distance(end1, end2) < tolerance)


def distance(point1, point2):
    """计算两个点之间的欧几里得距离"""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

#########################  笔画详细评价        #######################
def calculate_midpoint(line):
    """计算直线的中点"""
    x_coords = [point[0] for point in line]
    y_coords = [point[1] for point in line]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))


def calculate_length(line):
    """计算直线的长度"""
    length = 0
    for i in range(1, len(line)):
        length += math.sqrt((line[i][0] - line[i-1][0])**2 + (line[i][1] - line[i-1][1])**2)
    return length


def calculate_angle(line):
    """计算直线的倾斜角度"""
    dx = line[-1][0] - line[0][0]
    dy = line[-1][1] - line[0][1]
    return math.atan2(dy, dx) * 180 / math.pi


def check_intersection(line1, line2):
    """检查两条直线是否相交并计算夹角"""
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
    """对比两条直线"""
    print(f"笔画{stroke_index}对比:")

    # 计算中点
    midpoint1 = calculate_midpoint(line1)
    midpoint2 = calculate_midpoint(line2)

    # 判断中点是否在一条直线上
    on_same_line = math.isclose(midpoint1[1], midpoint2[1], abs_tol=1e-5)
    print(f"  - 临摹字笔画是否在模板字中点: {'是，笔画重心正确' if on_same_line else '否，笔画重心偏移'}")

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
        print(f"  - 笔画相交，夹角为: {angle_diff:.2f}°")
    else:
        if midpoint1[1] > midpoint2[1]:
            print("  - 临摹直线位于模板直线上方")
        else:
            print("  - 临摹直线位于模板直线下方")

    print("")  # 空行分隔每条笔画的结果

#########################  主流程        #######################
def main():
    handwritten_file_path = "E:\\python files\\Conda_env_files\\pythonProject1\\wordServer-main\\笔画坐标提取和评价\\bihua_data\\王.txt"  # 手写汉字坐标文件路径
    gif_file_path = "E:\\python files\\Conda_env_files\\pythonProject1\\wordServer-main\\笔画坐标提取和评价\\gif_data\\王.txt"  # GIF汉字坐标文件路径

    # 从文件中读取坐标
    # print("Reading handwritten coordinates...")
    handwritten_coords = read_coordinates_from_file(handwritten_file_path)
    # print("Reading GIF coordinates...")
    gif_coords = read_coordinates_from_file(gif_file_path)

    # 打印读取到的坐标
    # print("Handwritten coordinates:", handwritten_coords)
    # print("GIF coordinates:", gif_coords)

    # 归一化处理
    # print("Normalizing coordinates...")
    handwritten_coords_normalized = normalize_coordinates(handwritten_coords)
    gif_coords_normalized = normalize_coordinates(gif_coords)

    # 打印归一化后的坐标
    # print("Handwritten coordinates (normalized):", handwritten_coords_normalized)
    # print("GIF coordinates (normalized):", gif_coords_normalized)
    #
    # 分割笔画
    # print("Splitting strokes...")
    strokes1 = split_strokes(handwritten_coords_normalized)
    strokes2 = split_strokes(gif_coords_normalized)

    # # 打印分割后的笔画
    # print("Handwritten strokes:", strokes1)
    # print("GIF strokes:", strokes2)

    # 先判断笔画个数是否一致
    if len(strokes1) == len(strokes2):
        print(f"笔画个数一致，都是 {len(strokes1)} 个笔画。")

        # 比较笔画顺序
        # print("Comparing stroke order...")
        if compare_strokes(strokes1, strokes2):
            print("手写汉字的书写顺序与GIF汉字的顺序一致")

            # 进行详细的笔画评价
            # print("Evaluating strokes...")
            for index, (stroke1, stroke2) in enumerate(zip(strokes1, strokes2), start=1):
                compare_lines(stroke1, stroke2, index)
        else:
            print("手写汉字的书写顺序与GIF汉字的顺序不一致")
    else:
        print(f"笔画数量不一致。手写字有 {len(strokes1)} 个笔画，GIF汉字有 {len(strokes2)} 个笔画。")


if __name__ == "__main__":
    main()

