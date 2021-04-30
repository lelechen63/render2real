from PIL import Image
from parsing.faceeye_parsing import parsing, vis_parsing_maps
import numpy as np
import face_alignment
from parsing.eye_parsing.iris_detector import IrisDetector
import dlib
from parsing.model import BiSeNet
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
idet = IrisDetector()
idet.set_detector(fa)

n_classes = 19
facenet = BiSeNet(n_classes=n_classes)
facenet.cuda()
facenet.load_state_dict(torch.load('checkpoint/face_parsing.pth'))
facenet.eval()



base_p = '/raid/celong/FaceScape/ffhq_aligned_img'
_file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/frontface_list.pkl', "rb")
front_indx = pickle.load(_file)
ids =  os.listdir(base_p)
ids.sort()

for id_p in ids:
    current_p = os.path.join( base_p , id_p)
    front_idx = front_indx[id_p]
   
    for motion_p in os.listdir(current_p):
        current_p1 = os.path.join( current_p , motion_p)
        img_path = os.path.join( current_p1, front_idx + '.jpg')
        parsing_path = img_path.replace('fsmview_images', 'fsmview_landmarks')[:-4] +'_parsing.png'


        img_path = '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  
        image = Image.open(img_path)
        res = parsing(img_path, facenet, idet)
        print (res.shape)
        vis_parsing_maps(image, res, save_parsing_path='9_parsing.jpg', save_vis_path = 'imgs/9_vis.jpg' )