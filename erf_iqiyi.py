import cv2,time
import numpy as np


def main():
  name = './image.jpg'
  img = cv2.imread(name) ## as rgb
  h,w  =img.shape[:2]

  r = 3
  img_blur = cv2.boxFilter(img, -1, (r,r))

  # for rgb
  img_erf = np.zeros_like(img, np.uint8)
  s = time.time()
  for i in range(r//2, h-r//2):
      for j in range(r//2, w-r//2):
          block = img_blur[i-r//2:i+r//2+1, j-r//2:j+r//2+1, :]
          j_ = np.unravel_index(np.argmin(np.sum((block - 1.*img[i,j])**2, 2)), (r,r))
          img_erf[i,j] = img_blur[i-r//2+j_[0], j-r//2+j_[1]]
  e = time.time()
  print('time:', e-s)
  cv2.imwrite('./res-cat.png', np.hstack([img, img_blur, img_erf]))


if __name__ == '__main__':
  main()
