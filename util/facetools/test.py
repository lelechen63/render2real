from PIL import Image
from parsing.faceeye_parsing import parsing, vis_parsing_maps
import numpy as np
import face_alignment
from parsing.eye_parsing.iris_detector import IrisDetector
import dlib
from parsing.model import BiSeNet
import os
import pickle
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

        # img_path = '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  
        image = Image.open(img_path)
        # try:
            #  img_path[:-4] +'_front.png'
        res = parsing(image, facenet, idet, 'imgs/ggg.png')
        vis_parsing_maps(image, res, save_parsing_path=parsing_path, save_vis_path ='/raid/celong/FaceScape/tmp/tmp2/' + id_p +'_' + motion_p +'_' +front_idx +'.png' ) 
        # except:
        #     print (img_path)
        #     continue