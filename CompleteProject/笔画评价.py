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
