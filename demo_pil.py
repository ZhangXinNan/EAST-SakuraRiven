

from PIL import Image
# from PILLOW import Image
from cv_util import *

# f = '_tmp/img_1.jpg'
# f = 'D:\\data_md\\liuchenxing\\20191120_screen\\test_img\\166471be-e46e-4694-8d79-398d85147a2a.FpkMKRdYnZUi3ZNBT2DTNC8bGz7T.jpg'
f = 'D:\\data_md\\liuchenxing\\20191120_screen\\test_img\\1a68dc49-6371-410e-98ad-d1bf58a58562.1572226612081.jpg'
img_pil = Image.open(f)
img_cv = cv2.imread(f)

img_pil.save('demo_pil.png')
cv2.imwrite('demo_cv.png', img_cv)

img_pil2cv = pil2cv(img_pil)
cv2.imwrite('demo_pil2cv.png', img_pil2cv)
img_cv2pil = cv2pil(img_cv)
img_cv2pil.save("demo_cv2pil.png")


