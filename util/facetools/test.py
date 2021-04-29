from PIL import Image
from parsing.face_parsing import parsing, vis_parsing_maps

image = Image.open('imgs/9.jpg')

res = parsing(image)
vis_parsing_maps(image, res, save_im=True)