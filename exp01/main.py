import cv2
import pic_manipulate as pic
import Sobel
from data import originImage

if __name__ == '__main__':
    # Step 1 平移图像
    img = originImage.copy()
    res1 = pic.copyTree(img, 190, 133, 190, 530, 315, 295, 1.0, 0.9)
    pic.showImg(res1, "moved tree")
    pic.wait4close(res1)

    # Step 2 分离出蓝色通道, 并二值化
    bmin, bmax = 230, 255
    gmin, gmax = 120, 255
    [b, g, _] = cv2.split(originImage)  # 分离出蓝色通道
    (_, binaryBlue) = cv2.threshold(b, bmin, bmax, cv2.THRESH_BINARY_INV)  # 阈值反二值化
    (_, binaryGreen) = cv2.threshold(g, gmin, gmax, cv2.THRESH_BINARY_INV)  # 阈值反二值化
    binaryFinal = binaryBlue | binaryGreen
    binaryFinalXOR = binaryBlue ^ binaryGreen
    pic.showImg(binaryFinal, "Binary_Final")
    pic.showImg(binaryFinalXOR, "Binary_FinalXOR")
    pic.showImg(binaryGreen, "Binary_Green")
    pic.showImg(binaryBlue, "Binary_Blue")
    outLoc = 'output/binaryBlue' + '_' + str(bmin) + '_' + str(bmax) + '.png'
    cv2.imwrite(outLoc, binaryBlue)
    pic.wait4close(binaryBlue)

    # Step 3 利用 Sobel 算子二值化
    img = originImage.copy()
    img = Sobel.sobel(img)
    pic.showImg(img, "Sobel")
    cv2.imwrite('output/sobel_bin.png', img)
    pic.wait4close(img)

    # Step 4 平移图像时, 值为 0 的用原图
    img = originImage.copy()
    res2 = pic.copyTree_threshold(img, binaryFinal, 262, 133, 190, 530, 310, 295, 0.8, 0.8)
    cv2.imwrite(r'output/result.png', res2)
    pic.showImg(res2, "final")
    pic.wait4close(res2)
