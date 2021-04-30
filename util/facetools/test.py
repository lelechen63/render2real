from PIL import Image
from parsing.faceeye_parsing import parsing, vis_parsing_maps
import numpy as np
import face_alignment
from parsing.eye_parsing.iris_detector import IrisDetector
import dlib
from parsing.model import BiSeNet
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
idet = IrisDetector()
idet.set_detector(fa)
# cp='checkpoint/face_parsing.pth'

n_classes = 19
facenet = BiSeNet(n_classes=n_classes)
facenet.cuda()
facenet.load_state_dict(torch.load('parsing/checkpoint/face_parsing.pth'))
facenet.eval()

img_path = '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  
image = Image.open(img_path)
# landmark = np.load(img_path.replace('ffhq_aligned_img', 'fsmview_landmarks')[:-3] + 'npy').transpose(1,0)[:,::-1]
res = parsing(img_path)
print (res.shape)
vis_parsing_maps(image, res, save_parsing_path='9_parsing.jpg', save_vis_path = 'imgs/9_vis.jpg' )