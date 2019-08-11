import face_alignment 

from skimage import io 

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import collections



fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input = False)

print("Type:", type(fa))
input = io.imread('john_wick.jpg')

preds = fa.get_landmarks(input)

print("Type of preds", type(preds))
print("Len of preds", len(preds))  
#print(*preds, sep ="\n")  



