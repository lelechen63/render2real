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
import time, threading

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
idet = IrisDetector()
idet.set_detector(fa)

n_classes = 19
facenet = BiSeNet(n_classes=n_classes)
facenet.cuda()
facenet.load_state_dict(torch.load('checkpoint/face_parsing.pth'))
facenet.eval()

base_p = '/raid/celong/FaceScape/ffhq_aligned_img'
# _file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/validface_list.pkl', "rb")
# valid_all = pickle.load(_file)
total_ids =  os.listdir(base_p)
total_ids.sort()

def get_parsing_batch( ids ):
    for id_p in ids:
        current_p = os.path.join( base_p , id_p)
        for motion_p in os.listdir(current_p):
            current_p1 = os.path.join( current_p , motion_p)
            # valid_idxs = valid_all[id_p +'__' + motion_p]
            # img_path = '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  

            # debug 
            # img_path =  '/raid/celong/FaceScape/ffhq_aligned_img/1/14_sadness/33.jpg' 
            # image = Image.open(img_path)
            # res = parsing(image, facenet, idet, img_path[:-4] +'_mask.png')
            # parsing_path = img_path.replace('ffhq_aligned_img', 'fsmview_landmarks')[:-4] +'_parsing.png'
            # vis_parsing_maps(image, res, save_parsing_path=parsing_path, save_vis_path ='/raid/celong/FaceScape/tmp/tmp2/' + id_p +'_' + motion_p +'_' + '33.png' ) 
            # print (gg)
            for valid_f in range( len ( os.listdir( current_p1 ))):
            # for valid_f in valid_idxs:
                img_path = os.path.join( current_p1, str(valid_f) + '.jpg')
                # if os.path.exists(img_path[:-4] +'_mask.png'):
                #     print (img_path, '-----')
                #     continue
                print ('+++', img_path)
                parsing_path = img_path.replace('ffhq_aligned_img', 'fsmview_landmarks')[:-4] +'_parsing.png'
                # try:
                image = Image.open(img_path)
                res = parsing(image, facenet, idet, img_path[:-4] +'_mask.png')
                vis_parsing_maps(image, res, save_parsing_path=parsing_path)#, save_vis_path ='/raid/celong/FaceScape/tmp/tmp2/' + id_p +'_' + motion_p +'_' +str(valid_f_ +'.png' ) 
                print('---------')
                # except:
                #     print ('**********')
                #     print (img_path)
                #     continue
get_parsing_batch(total_ids)
# batch = 7
# for i in range(1):
#     threading.Thread(target = get_parsing_batch, args = (total_ids[batch * i: batch *(i+1)], )).start()
