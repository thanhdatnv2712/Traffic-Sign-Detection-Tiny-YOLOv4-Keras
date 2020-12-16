from yolo import YOLO
from PIL import Image

yolo = YOLO()

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         # r_image.show()
#         save
# yolo.close_session()
import time

path= "/home/ubuntu/01.Datasets/TAD16K/images/00001.jpg"
image= Image.open(path)
for i in range(10):
    #tic= time.time()
    r_image = yolo.detect_image(image)
#r_image.save("test.png")
yolo.close_session()
