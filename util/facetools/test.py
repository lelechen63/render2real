from PIL import Image
from parsing.face_parsing import parsing, vis_parsing_maps

image = Image.open('imgs/9.jpg')

res = parsing(image)
print (res.shape)
vis_parsing_maps(image, res, save_parsing_path='imgs/9_parsing.jpg', save_vis_path = 'imgs/9_vis.jpg' )