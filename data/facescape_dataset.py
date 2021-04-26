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
        self.dir_A = os.path.join(opt.dataroot, "fsmview_renderings")

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "fsmview_images")


        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/train.pkl"), "rb")
        else:
            _file = open(os.path.join(opt.dataroot, "lists/test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)
        _file.close()

        
    def __getitem__(self, index):        
        ### input A (renderred image)
        A_path = os.path.join( self.dir_A , data_list[index] )   
        print (A_path)          
        A = Image.open(A_path).convert('RGB')   
        params = get_params(self.opt, A.size)
        
        transform_A = get_transform(self.opt, params)      
        A_tensor = transform_A(A)


        B_tensor = 0
        ### input B (real images)
        B_path = os.path.join( self.dir_B , data_list[index] )   
        print (B_path)       
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)

     
        input_dict = { 'renderred_image':A_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeDataset'