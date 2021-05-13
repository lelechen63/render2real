import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import PIL
import json
import pickle 
import cv2
import numpy as np
import random
import torch
import trimesh, os
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
    angle_lists =  open("/raid/celong/lele/github/idinvert_pytorch/predef/angle_list2.txt", 'r')
    total_list = {}
    while True:
        line = angle_lists.readline()[:-1]
        if not line:
            break
        tmp = line.split(',')
        if tmp[0] +'/' + tmp[1] not in total_list.keys():

            total_list[tmp[0] +'/' + tmp[1] ]  = {}
        total_list[tmp[0] +'/' + tmp[1] ][tmp[2]] = [float(tmp[3]),float(tmp[4]), float(tmp[5])]

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

        ### json 
        self.dir_json = os.path.join(opt.dataroot, "fsmview_images")

        # /raid/celong/FaceScape/fsmview_landmarks/99/14_sadness/1_eye.png
        self.exp_set =  get_exp()

        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/img_alone_train.pkl"), "rb")
            
        else:
            _file = open(os.path.join(opt.dataroot, "lists/img_alone_test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)#[:1]
        _file.close()
        
        dic_file = open(os.path.join(opt.dataroot, "lists/img_dic_train.pkl"), "rb")
        self.dic_list = pickle.load(dic_file)#[:10]

        self.angle_list = get_anlge_list()
        
    def __getitem__(self, index):

        tmp = self.data_list[index].split('/')
        A_path = os.path.join( self.dir_A , self.data_list[index] ) 
        mask_path = os.path.join( self.dir_A , self.data_list[index][:-4] + '_mask.png' )
        json_path = os.path.join( self.dir_json , tmp[0], tmp[1], 'params.json' )
        
        f  = open(json_path , 'r')
        params = json.load(f)
        viewpoint = [np.array(params['%s_Rt' %  tmp[2][:-4]]).flatten()]
        ### input mask (binary mask to segment person out)
        mask = cv2.imread(mask_path)[:,:,::-1]
        ### input A (real image)
        A = cv2.imread(A_path)[:,:,::-1]
        A = A * mask
        A = Image.fromarray(np.uint8(A))
        params = get_params(self.opt, A.size)
        transform = get_transform(self.opt, params)      
        A_tensor = transform(A)

        small_index = 0
        A_angle = self.angle_list[tmp[0] +'/' + tmp[1]][tmp[2][:-4]]
        
        pid = tmp[0]
        expresison = tmp[1]

        # randomly get paired image (same identity or same expression)
        toss = random.getrandbits(1)
        # toss 0-> same iden, diff exp
        if toss == 0:
            pool = set(self.dic_list[pid].keys()) - set(expresison)
            B_exp = random.sample(pool, 1)[0]
            B_id = pid
            B_angle_pool = self.angle_list[pid +'/' + B_exp]
        # toss 1 -> same exp, diff iden
        else:
            pool = set(self.dic_list[expresison].keys()) - set(pid)
            B_id = random.sample(pool, 1)[0]
            B_exp = expresison
            B_angle_pool = self.angle_list[B_id +'/' + expresison]
        
        ggg = []
        for i in range(len(B_angle_pool)):
            ggg.append(B_angle_pool[str(i)])
        ggg = np.array(ggg)
        diff = abs(ggg - A_angle).sum(1)
        
        for kk in range(diff.shape[0]):
            small_index = diff.argsort()[kk]
            try:
                # print (small_index)
                B_path =  os.path.join( self.dir_A ,  B_id, B_exp, str(small_index) +'.jpg' )   
                # print (B_path)
                ### input mask (binary mask to segment person out)
                mask_path =os.path.join( self.dir_A ,B_id, B_exp, str(small_index)+ '_mask.png' )   
                # mask = Image.open(mask_path).convert('RGB')
                mask = cv2.imread(mask_path)[:,:,::-1] 
                B = cv2.imread(B_path)[:,:,::-1]
                break
            except:
                continue
        json_path = os.path.join( self.dir_json , B_id, B_exp, 'params.json' )
        f  = open(json_path , 'r')
        params = json.load(f)
        
        viewpoint.append(np.array(params['%d_Rt' %  small_index]).flatten())
        B = B * mask
        B = Image.fromarray(np.uint8(B))
        B_tensor = transform(B)
        viewpoint = np.asarray(viewpoint)
        viewpoint = torch.FloatTensor(viewpoint)

        input_dict = { 'image':A_tensor, 'pair_image': B_tensor, 'pair_type': toss, 'viewpoint' : viewpoint, 'A_path': self.data_list[index][:-4] , 'B_path': os.path.join(B_id, B_exp, str(small_index)) }

        return input_dict

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeDirDataset'


class FacescapeMeshTexDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        img_size = (1024,1024)

        ### input A (texture and mesh)
        self.dir_A = os.path.join(opt.dataroot, "textured_meshes")

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input C (eye parsing images)
        self.dir_C = os.path.join(opt.dataroot, "fsmview_landmarks")

        ### json 
        self.dir_json = os.path.join(opt.dataroot, "fsmview_images")

        self.exp_set =  get_exp()

        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/texmesh_train.pkl"), "rb")
            
        else:
            _file = open(os.path.join(opt.dataroot, "lists/texmesh_test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)#[:1]
        _file.close()
        
        ids = open(os.path.join(opt.dataroot, "lists/ids.pkl"), "rb")
        self.id_set = set(pickle.load(ids))
        self.exp_set = get_exp()
        self.facial_seg = cv2.imread("./predef/facial_mask_v10.png")[:,:,::-1]
        self.facial_seg = cv2.resize(self.facial_seg, img_size, interpolation = cv2.INTER_AREA)
    def __getitem__(self, index):

        tmp = self.data_list[index].split('/')
        # id_p , 'models_reg', motion_p
        # tex 
        tex_path = os.path.join( self.dir_A , self.data_list[index] + '.jpg')
        # mesh 
        tex = cv2.imread(tex_path)[:,:,::-1]
        tex = cv2.resize(tex, img_size, interpolation = cv2.INTER_AREA)
        tex = tex * self.facial_seg
        tex = Image.fromarray(np.uint8(tex))
        params = get_params(self.opt, tex.size)
        transform = get_transform(self.opt, params)      
        A_tex_tensor = transform(tex)

        print (A_tex_tensor.shape)
        mesh_path = os.path.join( self.dir_A , self.data_list[index] + '.obj')

        mesh = trimesh.load(mesh_path, process=False)
        vertices = mesh.vertices
        vertices=vertices.reshape(:, 4, 3)
        A_vertices = vertices[:, 0, :].reshape(-1)
        print( A_vertices.shape )

        toss = random.getrandbits(1)
        # toss 0-> same iden, diff exp
        if toss == 0:
            pool = self.exp_set - set(tmp[-1])
            B_exp = random.sample(pool, 1)[0]
            B_id = tmp[0]
        # toss 1 -> same exp, diff iden
        else:
            pool = self.id_set - set(tmp[0])
            B_id = random.sample(pool, 1)[0]
            B_exp = tmp[-1]
        
        # tex 
        tex_path = os.path.join( self.dir_A , B_id, 'models_reg' , B_exp,  + '.jpg')
        # mesh 
        tex = cv2.imread(tex_path)[:,:,::-1]
        tex = cv2.resize(tex, img_size, interpolation = cv2.INTER_AREA)
        tex = tex * self.facial_seg
        tex = Image.fromarray(np.uint8(tex))
        params = get_params(self.opt, tex.size)
        transform = get_transform(self.opt, params)      
        B_tex_tensor = transform(tex)

        print (B_tex_tensor.shape)
        tex_path = os.path.join( self.dir_A , B_id, 'models_reg' , B_exp,  + '.obj')
        mesh = trimesh.load(mesh_path, process=False)
        vertices = mesh.vertices
        vertices=vertices.reshape(:, 4, 3)
        B_vertices = vertices[:, 0, :].reshape(-1)
        print( B_vertices.shape)
        input_dict = { 'Atex':A_tex_tensor, 'Amesh': A_vertices, 'A_path': self.data_list[index], 'Btex':B_tex_tensor, 'Bmesh': B_vertices, 'B_path': os.path.join( B_id, 'models_reg' , B_exp), 'map_type':toss}

        return input_dict

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeMeshTexDataset'