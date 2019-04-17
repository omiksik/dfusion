import cv2
import numpy as np
import os

sparse_depth_input_dir = 'data/kitti_full/0008/lidar_sparse_depth'
dense_depth_input_dir  = 'data/kitti_full/0008/psm_depth_57'
output_dir 			   = 'data/kitti_full/0008/psm_depth_57_masked_erode'

# create output dir
if not os.path.exists(output_dir):
    print('Creating output dir: {}'.format(output_dir))
    os.makedirs(output_dir)

# get filenames
all_filenames = os.listdir(sparse_depth_input_dir)

# for each file
for filename in all_filenames:
	# filename = '000000.png'
	print('Processing image: {}'.format(filename))

	# open files
	sparse_lidar = cv2.imread(os.path.join(sparse_depth_input_dir, filename), cv2.IMREAD_GRAYSCALE)
	dense_lidar  = cv2.imread(os.path.join(dense_depth_input_dir, filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

	# find convex hull
	sparse_points = cv2.findNonZero(np.array(sparse_lidar, dtype = np.uint8))
	hull = cv2.convexHull(sparse_points)

	# create mask
	mask = np.zeros(dense_lidar.shape, np.uint8)
	cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1, 8)


	# kernel = np.ones((5,5),np.uint8)
	kernel = np.ones((5,1),np.uint8)
	erosion = cv2.erode(mask, kernel,iterations = 15)
	erosion[-100:, :] = mask[-100:, :] # 255

	# cv2.imshow('mask', mask)
	# cv2.imshow('mask2', erosion)

	# cv2.waitKey(0)
	mask = erosion

	# keep only valid part of the depth map
	masked_lidar = cv2.bitwise_and(dense_lidar, dense_lidar, mask = mask)

	# store result
	cv2.imwrite(os.path.join(output_dir, filename), masked_lidar)



























########################################
# tmp = cv2.imread('tmp.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

'''
img = cv2.imread('C:/Users/ondra/Documents/my_code/dense_multibody_slam_code/data/kitti_full/0001/lidar_sparse_depth/000000.png', cv2.IMREAD_GRAYSCALE)

n2 = cv2.findNonZero(np.array(img, dtype = np.uint8))

hull = cv2.convexHull(n2)

drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
cv2.drawContours(drawing, [hull], 0, (255, 255, 255), -1, 8)

dense_depth = cv2.imread('C:/Users/ondra/Documents/my_code/dense_multibody_slam_code/data/kitti_full/0001/lidar_dense_depth/test_output_epoch_39/000000.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
res = cv2.bitwise_and(dense_depth, dense_depth, mask = drawing)

cv2.imwrite('tmp.png', res)

'''



'''
import cv2
import numpy as np
import os


img = cv2.imread('C:/Users/ondra/Documents/my_code/dense_multibody_slam_code/data/kitti_full/0001/lidar_sparse_depth/000000.png', cv2.IMREAD_GRAYSCALE)


# retval, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

# cv2.imshow('threshold',threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# t2 = np.array(threshold, dtype = np.uint8)
# print(t2.shape)
# cv2.imshow('t2',t2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# n2 = cv2.findNonZero(t2)
# print(len(n2))
# #print(len(img[:, :, :] > 0.0))

n2 = cv2.findNonZero(np.array(img, dtype = np.uint8))
print(len(n2))

hull = cv2.convexHull(n2)
print(hull)

drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
cv2.drawContours(drawing, [hull], 0, (255, 255, 255), -1, 8)
print(np.amax(drawing))

cv2.imshow('contour',drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# complement = cv2.bitwise_not(drawing)
# cv2.imshow('complement',complement)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

dense_depth = cv2.imread('C:/Users/ondra/Documents/my_code/dense_multibody_slam_code/data/kitti_full/0001/lidar_dense_depth/test_output_epoch_39/000000.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
print(dense_depth.dtype)
print(np.amax(dense_depth))
res = cv2.bitwise_and(dense_depth, dense_depth, mask = drawing)
print(res.dtype)

cv2.imshow('masked',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('tmp.png', res)
tmp = cv2.imread('tmp.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
print(tmp.dtype)
print(np.amax(tmp))
'''