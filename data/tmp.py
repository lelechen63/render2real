from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL
import cv2
import numpy as np

facial_seg = Image.open("./predef/facial_mask_v10.png")
facial_seg  = np.array(facial_seg ) / 255.0
facial_seg = np.expand_dims(facial_seg, axis=2)
print (np.unique(facial_seg))

tex_path = '/raid/celong/FaceScape/textured_meshes/1/models_reg/16_grin.jpg'
# mesh 
tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
tex  = np.array(tex ) * facial_seg
tex = tex.astype(int)
print (np.unique(tex))
print (tex.shape)
tex = cv2.cvtColor(tex.astype(int), cv2.COLOR_RGB2BGR)
# tex = cv2.imread(tex_path)
# tex = cv2.resize(tex, self.img_size, interpolation = cv2.INTER_AREA)
# tex = tex * self.facial_seg
cv2.imwrite('./gg.png', tex)