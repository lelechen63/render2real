from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL
import cv2
import numpy as np

facial_seg = Image.open("./predef/facial_mask_v10.png")
facial_seg  = np.array(facial_seg )
kk = cv2.imread("./predef/facial_mask_v10.png")
gray = cv2.cvtColor(kk, cv2.COLOR_BGR2GRAY)

def bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

bbox = bbox(gray)
print (bbox)
y = bbox[0]
x = bbox[2]
h = bbox[1] - x
w = bbox[3] - y
# edged = cv2.Canny(gray, 30, 200)
# cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(edged, 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print (len(cnts), len(cnts[0]))
# cnts = cnts[0][0]
# x,y,w,h = cv2.boundingRect(cnts)
x = 1169-250
y =600-150
w =2000 
h = 1334

x = 1019
y =500
w =2000
h = 1334
l = max(w,h)

# print (x,y,w, h)

# 1169-150 600-150 2260 1334


facial_seg = facial_seg / 255.0
facial_seg = np.expand_dims(facial_seg, axis=2)
print (np.unique(facial_seg))


tex_path = '/raid/celong/FaceScape/texture_mapping/target/1/9_mouth_right.png'
# mesh 
tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
tex  = np.array(tex ) * facial_seg
tex =  np.uint8(tex)
l = max(w, h)
# kkk =  int(x - (l-w)/2)
# tex = tex[y:y+l,kkk :kkk +l,:]
tex = tex[y:y+l,x :x +l,:]
tex = cv2.cvtColor(tex, cv2.COLOR_RGB2BGR)

# tex = cv2.imread(tex_path)
# tex = cv2.resize(tex, self.img_size, interpolation = cv2.INTER_AREA)
# tex = tex * self.facial_seg
cv2.imwrite('./gg.png', tex)