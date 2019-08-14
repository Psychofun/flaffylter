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


filters = ['images/sunglasses.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png','images/pixel.png']
filterIndex = 6

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


    
# Main Function to apply dog filter
def add_filter(img_path, dog_filter):
    

    img = open_image_cv2(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces, preds = get_features_from_image(img)
    #print("PREDS",preds)


    if len(faces)==0:
        print("No faces Detected in the image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for x,y,w,h,_ in faces:   
        print("x y wh",x,y,w,h)  
        
        x,y,w,h = tuple(map(int, [x,y,w,h] ))
        # add bounding box to color image
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        gray_face = gray[y: y + h,x: x + w]
        color_face = img[y: y + h,x: x + w]
        

        eye1, eye2 = (preds[PRED_TYPES['eye1'].slice,0],preds[PRED_TYPES['eye1'].slice,1]),(preds[PRED_TYPES['eye2'].slice,0],preds[PRED_TYPES['eye1'].slice,1])

       

        
        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)

        y_coordinates = eye1[1] + eye2[1]

        max_y,min_y = np.max(y_coordinates), np.min(y_coordinates) 
        sunglass_width = int(  max_y  -  min_y  ) *2


        x_coordinates =  eye1[0] + eye2[0]
        max_x,min_x =np.max(x_coordinates) , np.min(x_coordinates)


        sunglass_heigth = int( max_x - min_x   ) *2
        print("Eye 1 {}, Eye 2 {}".format(eye1,eye2))


        print("Sunglasse width {} and height {} ".format(sunglass_width, sunglass_heigth )) 

        sunglass_resized = cv2.resize(sunglasses, (sunglass_heigth,sunglass_width),interpolation = cv2.INTER_CUBIC)
        transparent_region = sunglass_resized[:,:,:3] != 0 
        
        #Overlay sunglasses over img
        #face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
        x_,y_ = int(np.min(eye1[0])),int(np.min(eye1[1]))
        img[  y_ : y_ +sunglass_width,x_: x_ + sunglass_heigth:] = sunglass_resized[:,:,:3]
        
        #transparent_region_2 = sunglasses[:,:,:3] != 0 

        #alpha = sunglasses[:, :, 3] / 255.0
        #new_image = np.dstack((alpha * sunglasses[:,:,0], alpha * sunglasses[:,:,1],alpha * sunglasses[:,:,2]))
        #img[0:sunglasses.shape[0]  ,0:sunglasses.shape[1] ,:] = new_image


        #for point in preds:
        #    cv2.circle(img, tuple(point), 1, (255,255,255), 1)
        
        # Test anchor point for sunglasses
        #cv2.circle(img, (x_,y_), 2, (255,0,0),2)
    #cv2.imshow("Filters", img)


       
    #Change to RGB
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    result = add_filter(file_path, dog_filter)

    plt.axis('off')
    plt.imshow(result)
    plt.show()
    