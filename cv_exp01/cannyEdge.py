import cv2

import pic_manipulate as pic
import blue01 as blue
from data import originImage

image = originImage.copy()

# Step 1 平移图像
pic.copyTree(image, 190, 133, 190, 530, 315, 295, 1.0, 0.9)
pic.showImg("", image)
# 发现天空效果不理想

# Step 2 分离出蓝色通道, 并二值化
[b, _, _] = pic.splitChannels(originImage)
threshold_blue = blue.get_threshold_sky(b, 180)
pic.showImg("", threshold_blue)

# Step 3 平移图像时, 值为 0 的用原图
image = originImage.copy()
pic.copyTree_threshold(image, threshold_blue, 190, 133, 190, 530, 315, 295, 1.0, 0.9)
