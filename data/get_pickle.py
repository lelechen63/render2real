import pickle
import os
import random
def get_image_pickle():
    
    base_p = '/raid/celong/FaceScape/fsmview_renderings'
    save_p = '/raid/celong/FaceScape/ffhq_aligned_img'

    _file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/validface_list.pkl', "rb")
    valid_indx = pickle.load(_file)
    print (valid_indx.keys())

    train_list = []
    test_list = []

    ids =  os.listdir(base_p)
    ids.sort()

    for id_p in ids:
        current_p = os.path.join( base_p , id_p)
        save_p1 = os.path.join( save_p , id_p)
        all_motions = os.listdir(current_p)
        random.shuffle(all_motions)
        for k, motion_p in enumerate(all_motions):
            current_p1 = os.path.join( current_p , motion_p)
            save_p2 = os.path.join( save_p1 , motion_p)
            if id_p +'__' + motion_p not  in valid_indx.keys():
                print (id_p +'__' + motion_p)
                continue
            for cam_idx in valid_indx[ id_p +'__' + motion_p ]:
                img_p = os.path.join( save_p2, cam_idx + '.jpg')
                output_p = os.path.join( save_p2 ,cam_idx + '_render.png')
                parsing_p = img_p[:-4].replace('ffhq_aligned_img', 'fsmview_landmarks' ) + '_parsing.png'
                # print (img_p, output_p, parsing_p)
                if os.path.exists(img_p) and os.path.exists(output_p) and os.path.exists(parsing_p) :
                    # if id_p =='12':
                    print ( os.path.join( id_p , motion_p, cam_idx + '.jpg'))
                    if k < 17:
                        train_list.append( os.path.join( id_p , motion_p, cam_idx + '.jpg') )
                    else:
                        test_list.append( os.path.join( id_p , motion_p, cam_idx + '.jpg') )

                else:
                    print (img_p,'========')
                    continue
                # print ('gg')
    print (len(train_list))
    with open('/raid/celong/FaceScape/lists/img_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/raid/celong/FaceScape/lists/img_test.pkl', 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
get_image_pickle()