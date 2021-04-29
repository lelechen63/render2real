from PIL import Image
from parsing.faceeye_parsing import parsing, vis_parsing_maps
import numpy as np
img_path = '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  
image = Image.open(img_path)

res = parsing(img_path, np.load(img_path.replace('ffhq_aligned_img', 'fsmview_landmarks')[:-3] + 'npy'))
print (res.shape)
vis_parsing_maps(image, res, save_parsing_path='9_parsing.jpg', save_vis_path = 'imgs/9_vis.jpg' )