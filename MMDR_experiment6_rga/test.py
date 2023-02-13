import cv2
import torch
import numpy as np
import torch.nn as nn

image = cv2.imread(r'C:\Users\Svain\Desktop\test_image\face_1_1_01_1-0.jpg')
image = image / 255.0
image = np.array(image, dtype=np.float32)
image = np.expand_dims(image, axis=0)
b, c, h, w = image.shape()
# image = cv2.GaussianBlur(image, (3, 3), 0, 0)
bsz, tsz = 8, 2

# image_B = cv2.bilateralFilter(image, bsz, 125, 200)
#
# image_C = cv2.bilateralFilter(image, tsz, 125, 200) - image_B
#
# image_T = image - cv2.bilateralFilter(image, tsz, 125, 200)

# image_B = cv2.GaussianBlur(image, (bsz, bsz), 0, 0)
#
# image_C = cv2.GaussianBlur(image, (tsz, tsz), 0, 0) - image_B
#
# image_T = image - cv2.GaussianBlur(image, (tsz, tsz), 0, 0)

image_B = cv2.blur(image, (bsz, bsz))

image_C = cv2.blur(image, (tsz, tsz)) - image_B

image_T = image - cv2.blur(image, (tsz, tsz))

cv2.imwrite('test_B.jpg', image_B[0]*255)
cv2.imwrite('test_C.jpg', image_C[0]*15*255)
cv2.imwrite('test_T.jpg', image_T[0]*30*255)


# image = cv2.imread(r'C:\Users\Svain\Desktop\test_image\face_1_1_01_5-0.jpg')
# image = image / 255.0
# image = np.ones((256, 256, 3))
# image = torch.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
# conv1 = nn.Conv2d(3, 3, (3, 3), padding=1)
# w1 = torch.Tensor(np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])).expand(3, 3, 3, 3)
# conv1.weight = nn.Parameter(w1)
# image = conv1(image)
#
# conv2 = nn.Conv2d(3, 3, (3, 3), padding=1)
# w2 = torch.Tensor(np.array([[1/9] * 3 for i in range(3)])).expand(3, 3, 3, 3)
# conv2.weight = nn.Parameter(w1)
#
# conv3 = nn.Conv2d(3, 3, (9, 9), padding=1)
# w3 = torch.Tensor(np.array([[1/81] * 9 for i in range(9)])).expand(3, 3, 9, 9)
# conv3.weight = nn.Parameter(w1)
# image_B = conv3(image).detach()
#
# image_C = conv2(image) - image_B
#
# image_T = image - conv2(image)
#
# image_B = image_B[0].permute(1, 2, 0).data.numpy()
# image_C = image_C[0].permute(1, 2, 0).data.numpy()
# image_T = image_T[0].permute(1, 2, 0).data.numpy()
#
# cv2.imwrite('test_B.jpg', image_B*255)
# cv2.imwrite('test_C.jpg', image_C*15*255)
# cv2.imwrite('test_T.jpg', image_T*30*255)
