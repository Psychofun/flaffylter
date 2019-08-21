import torch
from filters import *
import sys
sys.path.append('./src')

from face_alignment_en import face_alignment_en


# To use syft with cuda.
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
import syft as sy 
hook = sy.TorchHook(torch)


NUM_WORKERS = 3
workers = [sy.VirtualWorker(hook, id = "w" + str(i)) for i in range(NUM_WORKERS) ]



if __name__ == '__main__':
    device_str = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device_str)
    
    file_path = 'john_wick.jpg'
    image = open_image_cv2(file_path)
    
    # Run the 2D face alignment on a test image, with CUDA.
    fa = face_alignment_en.FaceAlignment(face_alignment_en.LandmarksType._2D, device= device_str, flip_input=True, workers = workers)
    #bounding boxes of faces found in image

    # Drop alpha channel.
    input_image = image[..., ::-1].copy()
    # Convert to tensor.
    input_image = torch.from_numpy(input_image).float().to(torch.device(device_str))
    # Encrypt input image
    input_image = input_image.fix_precision().share(*workers)




    boxes_faces = fa.face_detector.detect_from_image( input_image )

    preds = fa.get_landmarks_from_image(image,detected_faces=boxes_faces)[-1]
    
    print("Box faces {} and  preds {}".format(boxes_faces, preds))
   
   
    

