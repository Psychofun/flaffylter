import os
import cv2
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False, workers  = []):
        super(SFDDetector, self).__init__(device, verbose)

        self.workers = workers

        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

        


    def detect_from_image(self, tensor_or_path):
        image = tensor_or_path.copy()
        # Interchange Last channel with first. C0,C1,C2 to C2,C1,C0
        first_channel = image[:,:,0].copy()
        image[:,:,0] = image[:,:,2]
        image[:,:,2] = first_channel

        bboxlist = detect(self.face_detector, image, device=self.device, workers = self.workers)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
