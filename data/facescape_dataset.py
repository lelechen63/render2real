import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pickle 

class FacescapeDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (renderred image)
        self.dir_A = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input C (eye parsing images)
        self.dir_C = os.path.join(opt.dataroot, "ffhq_aligned_img")
        # /raid/celong/FaceScape/fsmview_landmarks/99/14_sadness/1_eye.png


        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/img_train.pkl"), "rb")
        else:
            _file = open(os.path.join(opt.dataroot, "lists/test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)[:10]
        _file.close()

        
    def __getitem__(self, index):        
        ### input A (renderred image)
        A_path = os.path.join( self.dir_A , self.data_list[index][:-4] + '_render.png' )   
          
        #for debug
        # A_path =  '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1_render.png'    
        # print (A_path)  
        A = Image.open(A_path).convert('RGB')   
        params = get_params(self.opt, A.size)
        
        transform = get_transform(self.opt, params)      
        A_tensor = transform(A)

        B_tensor = 0
        ### input B (real images)
        B_path = os.path.join( self.dir_B , self.data_list[index] )   
        #for debug
        # B_path =  '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1.jpg'  
        # print (B_path)       
        B = Image.open(B_path).convert('RGB')
        # transform_B = get_transform(self.opt, params)      
        B_tensor = transform(B)

        C_path =  os.path.join( self.dir_A , self.data_list[index][:-4] + '_parsing.png' )
        #debug 
        # C_path =  '/raid/celong/FaceScape/ffhq_aligned_img/1/1_neutral/1_parsing.png'    

        C =  Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

     
        input_dict = { 'renderred_image':A_tensor, 'image': B_tensor, 'eye_parsing': C_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeDataset'