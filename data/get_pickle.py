import pickle
import os
import random
import openmesh
import PIL
def get_image_pickle():
    
    base_p = '/raid/celong/FaceScape/ffhq_aligned_img'
    save_p = '/raid/celong/FaceScape/ffhq_aligned_img'

    _file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/validface_list.pkl', "rb")
    valid_indx = pickle.load(_file)
    print(len(valid_indx))
    # print (valid_indx.keys())

    hhh = 0
    train_list = []
    test_list = []

    ids =  os.listdir(base_p)
    ids.sort()
    invalid = []
    total = 0
    for id_p in ids:
        current_p = os.path.join( base_p , id_p)
        save_p1 = os.path.join( save_p , id_p)
        all_motions = os.listdir(current_p)
        random.shuffle(all_motions)
        for k, motion_p in enumerate(all_motions):
            current_p1 = os.path.join( current_p , motion_p)
            save_p2 = os.path.join( save_p1 , motion_p)
            if id_p +'__' + motion_p not  in valid_indx.keys():
                continue
            for cam_idx in valid_indx[ id_p +'__' + motion_p ]:
                total +=1
                img_p = os.path.join( save_p2, cam_idx + '.jpg')
                output_p = os.path.join( save_p2 ,cam_idx + '_render.png')
                parsing_p = img_p[:-4].replace('ffhq_aligned_img', 'fsmview_landmarks' ) + '_parsing.png'
                # print (img_p, output_p, parsing_p)
                # if os.path.exists(img_p) and os.path.exists(output_p) and os.path.exists(parsing_p) :
                if os.path.exists(img_p)  and os.path.exists(parsing_p) :
                    # if id_p =='12':
                    # print ( os.path.join( id_p , motion_p, cam_idx + '.jpg'))
                    if k < 17:
                        train_list.append( os.path.join( id_p , motion_p, cam_idx + '.jpg') )
                    else:
                        test_list.append( os.path.join( id_p , motion_p, cam_idx + '.jpg') )

                else:
                    # print (img_p, parsing_p)
                    invalid.append(parsing_p)
                    continue
                # print ('gg')
    print (len(train_list), len(test_list), total,len(invalid))

    with open('/raid/celong/FaceScape/lists/img_alone_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/raid/celong/FaceScape/lists/img_alone_test.pkl', 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_paired_image_pickle():
    _file = open(os.path.join('/raid/celong/FaceScape', "lists/img_alone_train.pkl"), "rb")
    all_train_list = pickle.load(_file)
    _file.close()
    train_list  = {}

    for item in all_train_list:
        tmp = item.split('/')
        pid = tmp[0]
        motion = tmp[1]
        if pid not in train_list.keys():
            train_list[pid] = {}
            train_list[pid][motion] = [item]
        else:
            if motion not in train_list[pid].keys():
                train_list[pid][motion] = [item]
            else:
                train_list[pid][motion].append(item)
        
        if motion not in train_list.keys():
            train_list[motion] = {}
            train_list[motion][pid] =[item]
        else:
            if pid not in train_list[motion].keys():
                train_list[motion][pid] = [item]
            else:
                train_list[motion][pid].append(item)

    print (len(train_list), len(train_list[motion]), len(train_list[motion][pid]))
    print (train_list[motion][pid])

    print (len(train_list), len(train_list[pid]), len(train_list[pid][motion]))
    print (train_list[pid][motion])
    
    with open('/raid/celong/FaceScape/lists/img_dic_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def get_texmesh_pickle():
    
    base_p = '/raid/celong/FaceScape/textured_meshes'

    train_list = []
    test_list = []
    ids =  os.listdir(base_p)
    ids.sort()
    for id_p in ids:
        current_p = os.path.join( base_p , id_p, 'models_reg')
        all_files = os.listdir(current_p)
        all_motions = []
        for f  in all_files:
            if 'jpg' in f:
                all_motions.append(f[:-4])
        random.shuffle(all_motions)
        for k, motion_p in enumerate(all_motions):
            # try:
                tex_path = os.path.join(current_p, motion_p + '.jpg')
                mesh_path = os.path.join(current_p, motion_p + '.obj')
                tex = PIL.Image.open(tex_path)
                om_mesh = openmesh.read_trimesh(mesh_path)
                A_vertices = np.array(om_mesh.points())
                print (A_vertices.shape)
                if A_vertices.shape[0] == 0:
                    continue
                if k < 17:
                    train_list.append( os.path.join( id_p , 'models_reg', motion_p) )
                else:
                    test_list.append( os.path.join( id_p , 'models_reg',  motion_p) )
            # except:
            #     continue
    print (test_list[:10])
    print (len(train_list), len(test_list))

    with open('/raid/celong/FaceScape/lists/texmesh_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/raid/celong/FaceScape/lists/texmesh_train.pkl', 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_paired_texmesh_pickle():
    base_p = '/raid/celong/FaceScape/textured_meshes'

    ids =  os.listdir(base_p)
    ids.sort()

    with open('/raid/celong/FaceScape/lists/ids.pkl', 'wb') as handle:
        pickle.dump(ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
# get_paired_texmesh_pickle()
get_texmesh_pickle()
# get_paired_image_pickle()
