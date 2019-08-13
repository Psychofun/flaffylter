import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

import collections
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt




def open_image(image_path):
    
    """
    image_path: string 
        path to image
    """

    try:
        input_img = io.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        

    return input_img
def open_image_cv2(image_path):
    """
    image_path: string 
        path to image
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError("File {} not found.".format(image_path))
    img = cv2.imread(image_path)

    return img

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)



# Overlay a transparent image on background.
def transparentOverlay(src , overlay , pos=(0,0),scale = 1):
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src    
    
def get_filter_spots(preds, scale_x, scale_y, scale, dog_filter):    
    #scale_x and y for scale of 96x96 image and scale for  overall filter scaling
    filter_nose = dog_filter['nose']
    filter_right_ear = dog_filter['ear_right']

    #Hyper-Paramaters
    y_padding = 5
    ear_padding = 6



    #Add nose
    nose_x = preds[PRED_TYPES['nose'].slice,0][1]
    nose_y = preds[PRED_TYPES['nose'].slice,1][1]

    print("Nose x {} and y {}".format(nose_x,nose_y))

    #exit()
    nose_x = int(nose_x - filter_nose.shape[1]*scale/2)
    nose_y = int( (nose_y + y_padding)*scale_y - filter_nose.shape[0]*scale/2)
    

    #result = transparentOverlay(img_crop.copy(),filter_nose,(x,y),scale)
    
    left_ear_x = 0 - ear_padding
    left_ear_y = 0 - ear_padding*2
    
    right_ear_x = int( (ear_padding*2)*scale_x - filter_right_ear.shape[0]*scale)
    right_ear_y = (0 - ear_padding)*scale_y
    
    return [nose_x, nose_y],[left_ear_x*scale_x, left_ear_y*scale_y],[right_ear_x, right_ear_y]

def get_best_scaling(w):
    filter_width = 420
    return 1.1*(w/filter_width)
    
def get_eye_angle(pred):
    left = pred[0:1]
    right = pred[2:3]
    #find vector
    vec = [right[0]-left[0], right[1]-left[1] ]
    angle = vec[0] / (math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]))
    
    return angle
    
# Main Function to apply dog filter
def add_dog_filter(img_path, dog_filter):
    

    img = open_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces, preds = get_features_from_image(img)



    filter_nose = dog_filter['nose']
    filter_left_ear = dog_filter['ear_left']
    filter_right_ear = dog_filter['ear_right']
    
    if len(faces)==0:
        print("No faces Detected in the image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for x,y,w,h,_ in faces:   
        print("x y wh",x,y,w,h)  
        
        x,y,w,h = tuple(map(int, [x,y,w,h] ))
        # add bounding box to color image
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        img_crop = img[y:y+h, x:x+w]
        
        scale_x = 1 #img_crop.shape[0]/img.shape[0]
        scale_y = 1 #img_crop.shape[1]/img.shape[1]
        print("Scale x {}, y {}".format(scale_x, scale_y))
        
        scale = get_best_scaling(w)
        print("Scale", scale)
        
        nose,left_ear,right_ear = get_filter_spots(preds, scale_x, scale_y, scale, dog_filter)  
        #nose, left_ear,rigth_ear = 

        #Add images
        result = transparentOverlay(img.copy(),filter_nose,( int(nose[0]+x), int(nose[1]+y)), scale)
        result = transparentOverlay(result.copy(),filter_left_ear,( int(left_ear[0]+x), int(left_ear[1]+y)), scale)
        result = transparentOverlay(result.copy(),filter_right_ear,( int(right_ear[0]+x), int(right_ear[1]+y)), scale)
        
        img = result
    #Change to RGB
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #and return
    return result
    
  

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
PRED_TYPES = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
}

"""
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('3d_landmarks_alone.png', bbox_inches=extent)
"""

def get_features_from_image(image):
    """
    image: array 
        image object .
    return boxes_faces (bounding boxes for faces), predictions (facial landmarks)
    """

   
    # Run the 2D face alignment on a test image, with CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
    #bounding boxes of faces found in image

    boxes_faces = fa.face_detector.detect_from_image(image[..., ::-1].copy())
    preds = fa.get_landmarks_from_image(image,detected_faces=boxes_faces)[-1]
    
   
    return boxes_faces, preds




def image_2d_landmarks(image_path):
    """
    image_path: string
        path to image.

    """

    try:
        input_img = io.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        

    boxes_faces,preds= get_features_from_image(input_img )

    print("Boxes_faces type", type(boxes_faces), boxes_faces,boxes_faces[0].shape)



    # 2D-Plot
    plot_style = dict(marker='o',
    markersize=4,
    linestyle='-',
    lw=2)

    fig = plt.figure(
                     #figsize = (10.24,10.24)
                     figsize=plt.figaspect(.5)
                    )
    ax = fig.add_subplot(1,1,1) #(1, 2, 1)
    ax.imshow(input_img)

    for pred_type in PRED_TYPES.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    fig.savefig('2d_landmarks.png')








def image_3d_landmarks():
    pass


if __name__ == "__main__":

    filter_path = './assets/filter3.png'
    filter_full = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED) #Read  PNG 

    
        
    dog_filter = {  'nose' : filter_full[302:390,147:300],
                    'ear_left' : filter_full[55:195,0:160],
                    'ear_right' : filter_full[55:190,255:420],
                }

    file_path = 'john_wick.jpg'
    #image_2d_landmarks(file_path)
    result = add_dog_filter(file_path, dog_filter)

    plt.axis('off')
    plt.imshow(result)
    plt.show()
    #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('Images/Edited/'+img_path[7:] ,result)
