
import face_alignment 
import torch
# To use syft with cuda.
"""torch.set_default_tensor_type(torch.cuda.FloatTensor)"""
import syft as sy 
hook = sy.TorchHook(torch)

from filters import *




NUM_WORKERS = 3
workers = [sy.VirtualWorker(hook, id = "w" + str(i)) for i in range(NUM_WORKERS) ]


class EncryptedFA:

    def __init__(self, landmarks_type,
                 device='cuda',
                 flip_input=False, workers = []):

        self.face_alignment = face_alignment.FaceAlignment(landmarks_type = landmarks_type,
                            device = device, flip_input = flip_input )
        

        # Encrypt face detector model.
        self.face_alignment.face_detector.face_detector = self.face_alignment.face_detector.face_detector.fix_precision().share(*workers)
        #self.face_detector = self.face_alignment.face_detector
        #print("Face detector type: ", self.face_alignment.face_detector.face_detector, type(self.face_alignment.face_detector.face_detector))
        #print("Face detector: ",list(self.face_alignment.face_detector.face_detector.parameters()))


        
        
        # Encrypt face alignment model.
        self.face_alignment.face_alignment_net = self.face_alignment.face_alignment_net.fix_precision().share(*workers)
        #Check parameters
        #print("Face alignment params: ",list(self.face_alignment.face_alignment_net.parameters()))

    def  get_landmarks_from_image(self, image_or_path, detected_faces=None):


        """Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.
         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        
       
        if detected_faces is None:
            detected_faces = self.face_alignment.face_detector.detect_from_image(image[..., ::-1].copy())

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        torch.set_grad_enabled(False)
        landmarks = []
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_alignment.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.face_alignment.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment.face_alignment_net(inp)[-1].detach()
            if self.face_alignment.flip_input:
                out += flip(self.face_alignment.face_alignment_net(flip(inp))
                            [-1].detach(), is_label=True)
            out = out.cpu()

            pts, pts_img = get_preds_fromhm(out, center, scale)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            if self.face_alignment.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.face_alignment.device)
                depth_pred = self.face_alignment.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())

        return landmarks




if __name__ == '__main__':
    fa = EncryptedFA(face_alignment.LandmarksType._2D, device='cpu', flip_input=True)


    file_path = 'john_wick.jpg'
    image = open_image_cv2(file_path)
    # Run the 2D face alignment on a test image, with CUDA.
    #bounding boxes of faces found in image

    input_image = torch.tensor(image[..., ::-1].copy()).fix_precision().share(*workers)
    boxes_faces = fa.face_alignment.face_detector.detect_from_image(input_image)
    preds = fa.get_landmarks_from_image(image[...,:3],detected_faces=boxes_faces)[-1]

    print("BOXES {}  \n Preds {}".format(boxes_faces, preds))
    
   




