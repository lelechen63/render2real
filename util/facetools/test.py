from PIL import Image
from parsing.faceeye_parsing import parsing, vis_parsing_maps

img_path = '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  
image = Image.open(img_path)

res = parsing(image)
print (res.shape)
vis_parsing_maps(image, res, save_parsing_path=img_path[:-4] + '_parsing.jpg', save_vis_path = img_path[:-4] +'_vis.jpg' )