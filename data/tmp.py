from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL
import cv2
tex_path = '/raid/celong/FaceScape/textured_meshes/1/models_reg/16_grin.jpg'
# mesh 
tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
tex  = np.array(tex ) 
tex = cv2.cvtColor(im_cv, cv2.COLOR_RGB2BGR)

# tex = cv2.resize(tex, self.img_size, interpolation = cv2.INTER_AREA)
# tex = tex * self.facial_seg
cv2.imwrite('./gg.png', tex)