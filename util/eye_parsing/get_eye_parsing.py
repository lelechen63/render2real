import cv2
from matplotlib import pyplot as plt
import face_alignment
from iris_detector import IrisDetector
import dlib
import numpy as np 
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/raid/celong/lele/github/idinvert_pytorch/utils/shape_predictor_68_face_landmarks.dat')

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


# def resize_image(im, max_size=768):
#     if np.max(im.shape) > max_size:
#         ratio = max_size / np.max(im.shape)呢
#         print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))})  在好  
root = '/raid/celong/FaceScape/'


def get_imgs(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def get_imgs():

    _file = open(os.path.join(root, "lists/train.pkl"), "rb")
    data_list = pickle.load(_file)
    _file.close()

    # _file = open(os.path.join(root, "lists/test.pkl"), "rb")
    # data_list.extend(pickle.load(_file))
    # _file.close()

    return data_list

dir_B = os.path.join(root , "fsmview_images")
data_list = get_imgs()
idet = IrisDetector()
idet.set_detector(fa)
for data in data_list:
    print (data)
    img_path = os.path.join(dir_B, data)
    print (img_path)


    im = cv2.imread(img_path)[..., ::-1]
    # im = resize_image(im) # Resize image to prevent GPU OOM.
    h, w, _ = im.shape
    eye_lms = idet.detect_iris(im)

    # Display detection result
    plt.figure(figsize=(15,10))
    draw = idet.draw_pupil(im, eye_lms[0][0,...]) # draw left eye
    draw = idet.draw_pupil(draw, eye_lms[0][1,...]) # draw right eye

    blank_image = np.zeros((h,w,3), np.uint8)
    lms =   eye_lms[0][0,...].astype(np.int32)[:,::-1]

    cv2.fillConvexPoly(blank_image, lms[:8], (0,0,255))
    cv2.fillConvexPoly(blank_image, lms[8:16], (255,0,0))

    lms =   eye_lms[0][1,...].astype(np.int32)[:,::-1]
    cv2.fillConvexPoly(blank_image, lms[:8], (0,0,255))
    cv2.fillConvexPoly(blank_image, lms[8:16], (255,0,0))
    cv2.imwrite('gg.png', blank_image)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    bbox = detector(gray, 1)[0]

    x0 = int(bbox.left())
    x1 = int(bbox.right())
    y0 = int(bbox.top())
    y1 = int(bbox.bottom())

    plt.subplot(1,2,1)
    plt.imshow(draw)
    plt.subplot(1,2,2)
    plt.imshow(draw[x0:x1, y0:y1])
    plt.savefig('foo.png')
    break