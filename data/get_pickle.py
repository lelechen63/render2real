import pickle

def get_image_pickle():
    ### input A (renderred image)
    self.dir_A = os.path.join(opt.dataroot, "ffhq_aligned_img")

    ### input B (real images)
    self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

    ### input C (eye parsing images)
    self.dir_C = os.path.join(opt.dataroot, "ffhq_aligned_img")

    base_p = '/raid/celong/FaceScape/fsmview_renderings'
    save_p = '/raid/celong/FaceScape/ffhq_aligned_img'

    _file = open( './predef/frontface_list.pkl', "rb")
    front_indx = pickle.load(_file)

    train_list = []

    ids =  os.listdir(base_p)
    ids.sort()

    for id_p in ids[K * 5: (K + 1) * 5]:
        current_p = os.path.join( base_p , id_p)
        save_p1 = os.path.join( save_p , id_p)
        front_idx = front_indx[id_p]
      
        for motion_p in os.listdir(current_p):
            current_p1 = os.path.join( current_p , motion_p)
            save_p2 = os.path.join( save_p1 , motion_p)
            
            img_p = os.path.join( save_p2, front_idx + '.jpg')
            output_p = os.path.join( save_p2 ,front_idx + '_render.png')
            parsing_p = img_p[:-4] + '_parsing.png'

            print (img_p, output_p, parsing_p)
            if os.path.exists(img_p) and os.path.exists(output_p) and os.path.exists(parsing_p) :
            
                train_list.append( os.path.join( current_p , motion_p, front_idx + '.jpg') )
    with open('/raid/celong/FaceScape/lists/img_train.pkl'), 'wb') as handle:
    pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)