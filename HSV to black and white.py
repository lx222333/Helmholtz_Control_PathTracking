import cv2
import numpy as np


def extract_red_to_white(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 调整这些参数以获得更好的红色提取效果
    sensitivity = 10  # 调整红色检测的灵敏度
    min_saturation = 40 # 最小饱和度
    min_value = 40  # 最小亮度

    # 红色的HSV范围（更宽松的设置）
    lower_red1 = np.array([0, min_saturation, min_value])
    upper_red1 = np.array([sensitivity, 255, 255])
    lower_red2 = np.array([180 - sensitivity, min_saturation, min_value])
    upper_red2 = np.array([180, 255, 255])

    # 创建掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 创建结果：黑色背景 + 白色红色区域
    result = np.zeros_like(img)
    result[red_mask > 0] = [255, 255, 255]


    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


# 使用函数
result = extract_red_to_white(r"D:\Helmhots\RRT\real_blood_vessels_picture\0000.png")