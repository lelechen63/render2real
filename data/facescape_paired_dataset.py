import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import pickle 
import cv2
import numpy as np
import random

def get_exp():
    expressions = {
        1: "1_neutral",
        2: "2_smile",
        3: "3_mouth_stretch",
        4: "4_anger",
        5: "5_jaw_left",
        6: "6_jaw_right",
        7: "7_jaw_forward",
        8: "8_mouth_left",
        9: "9_mouth_right",
        10: "10_dimpler",
        11: "11_chin_raiser",
        12: "12_lip_puckerer",
        13: "13_lip_funneler",
        14: "14_sadness",
        15: "15_lip_roll",
        16: "16_grin",
        17: "17_cheek_blowing",
        18: "18_eye_closed",
        19: "19_brow_raiser",
        20: "20_brow_lower"
    }
    exps = []
    for i in range(1,21):
        exps.append(expressions[i])
    return set(exps)
def get_anlge_list():
    angle_lists =  open("/raid/celong/lele/github/idinvert_pytorch/predef/angle_list.txt", 'r')
    total_list = {}
    while True:
        line = angle_lists.readline()[:-1]
        if not line:
            break
        tmp = line.split(',')
        if tmp[0] +'/' + tmp[1] not in total_list.keys():

            total_list[tmp[0] +'/' + tmp[1] ]  = {}
        total_list[tmp[0] +'/' + tmp[1] ][tmp[2]] = [float(tmp[3]),float(tmp[4]), float(tmp[5])]
    # print (len(total_list))

    return total_list
class FacescapeDirDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (renderred image)
        self.dir_A = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input C (eye parsing images)
        self.dir_C = os.path.join(opt.dataroot, "fsmview_landmarks")
        # /raid/celong/FaceScape/fsmview_landmarks/99/14_sadness/1_eye.png
        self.exp_set =  get_exp()

        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/img_alone_train.pkl"), "rb")
            
        else:
            _file = open(os.path.join(opt.dataroot, "lists/img_alone_test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)
        _file.close()
        
        dic_file = open(os.path.join(opt.dataroot, "lists/img_dic_train.pkl"), "rb")
        self.dic_list = pickle.load(dic_file)

        self.angle_list = get_anlge_list()
        
    def __getitem__(self, index):

        ### input mask (binary mask to segment person out)
        mask_path =os.path.join( self.dir_A , self.data_list[index][:-4] + '_mask.png' )   
        # mask = Image.open(mask_path).convert('RGB')
        mask = cv2.imread(mask_path)[:,:,::-1]
        ### input A (real image)
        A_path = os.path.join( self.dir_A , self.data_list[index] )   
        
        #for debug
        # A_path =  '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'    
        A = cv2.imread(A_path)[:,:,::-1]
        A = A * mask
        A = Image.fromarray(np.uint8(A))
        params = get_params(self.opt, A.size)
        transform = get_transform(self.opt, params)      
        A_tensor = transform(A)

        # randomly get paired image (same identity or same expression)
        
        tmp = self.data_list[index].split('/')
        # print ( self.angle_list[tmp[0] +'/' + tmp[1]].keys())
        A_angle = self.angle_list[tmp[0] +'/' + tmp[1]][tmp[2][:-4]]
        # print (A_angle)
        viewpoint = [A_angle]
        pid = tmp[0]
        expresison = tmp[1]

        toss = random.getrandbits(1)
        # toss 0-> same iden, diff exp
        if toss == 0:
            pool = set(self.dic_list[pid].keys()) - set(expresison)
            B_exp = random.sample(pool, 1)[0]
            B_id = pid
            B_angle_pool = self.angle_list[pid +'/' + B_exp]
            # print (B_angle_pool)

        # toss 1 -> same exp, diff iden
        else:
            pool = set(self.dic_list[expresison].keys()) - set(pid)
            B_id = random.sample(pool, 1)[0]
            B_exp = expresison
            B_angle_pool = self.angle_list[B_id +'/' + expresison]
            # print (B_angle_pool)
        
        tmp = []
        for i in range(len(B_angle_pool)):
            tmp.append(B_angle_pool[str(i)])
        tmp = np.array(tmp)

        diff = (tmp - A_angle).sum(1)
        diff = diff.argsort()
        for kk in range(diff.shape[0]):
            small_index = diff[kk]
            try:
                # print (small_index)
                B_path =  os.path.join( self.dir_A ,  B_id, B_exp, str(small_index) +'.jpg' )   
                # print (B_path)

                ### input mask (binary mask to segment person out)
                mask_path =os.path.join( self.dir_A ,B_id, B_exp, str(small_index)+ '_mask.png' )   
                # mask = Image.open(mask_path).convert('RGB')
                mask = cv2.imread(mask_path)[:,:,::-1]
            
                #for debug
                # B_path =  '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'    
                B = cv2.imread(B_path)[:,:,::-1]
                viewpoint.append(tmp[str(small_index)])
                break
            except:
                continue
        B = B * mask
        B = Image.fromarray(np.uint8(B))
        params = get_params(self.opt, B.size)
        transform = get_transform(self.opt, params)      
        B_tensor = transform(B)


        # B_tensor =1
    
        input_dict = { 'image':A_tensor, 'pair_image': B_tensor, 'pair_type': toss, 'viewpoint' : viewpoint 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeDirDataset'