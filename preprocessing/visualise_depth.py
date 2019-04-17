import cv2
import numpy as np

input_image  = 'data/vkitti_dataset/depth/00370.png'
output_image = 'data/vkitti_dataset/00370.png'

depth = cv2.imread(input_image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

threshold = 4000.0

depth[depth >= threshold] = 0

print(depth.min())
print(depth.max())

depth2 = depth.astype(np.float) #  / 40000.0
depth2 /= threshold
depth2 *= 255.0

depth2 = depth2.astype(np.uint8)

print(depth2.min())
print(depth2.max())

im_color = cv2.applyColorMap(depth2, cv2.COLORMAP_JET)

cv2.imwrite(output_image, im_color)