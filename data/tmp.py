from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL
import cv2
import numpy as np

facial_seg = Image.open("./predef/facial_mask_v10.png")
facial_seg  = np.array(facial_seg )
kk = cv2.imread("./predef/facial_mask_v10.png")
gray = cv2.cvtColor(kk, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0]
x,y,w,h = cv2.boundingRect(cnts)

print (x,y,w, h)


facial_seg = facial_seg / 255.0
facial_seg = np.expand_dims(facial_seg, axis=2)
print (np.unique(facial_seg))


tex_path = '/raid/celong/FaceScape/textured_meshes/1/models_reg/16_grin.jpg'
# mesh 
tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
tex  = np.array(tex ) * facial_seg
tex =  np.uint8(tex)
print (np.unique(tex))
print (tex.shape)
tex = tex[y:y+max(h,w),x:x+max(h,w),:]
tex = cv2.cvtColor(tex, cv2.COLOR_RGB2BGR)

# tex = cv2.imread(tex_path)
# tex = cv2.resize(tex, self.img_size, interpolation = cv2.INTER_AREA)
# tex = tex * self.facial_seg
cv2.imwrite('./gg.png', tex)