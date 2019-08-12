import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections





# 3D-Plot
"""ax = fig.add_subplot(1,1,1,projection = '3d')   #add_subplot(1, 2, 2, projection='3d')
surf = ax.scatter(preds[:, 0] * 1.2,
                  preds[:, 1],
                  preds[:, 2],
                  c='cyan',
                  alpha=1.0,
                  edgecolor='b')

for pred_type in pred_types.values():
    ax.plot3D(preds[pred_type.slice, 0] * 1.2,
              preds[pred_type.slice, 1],
              preds[pred_type.slice, 2], color='blue')

ax.view_init(elev=90., azim=90.)
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
"""
# Save Face
#extent = ax.get_window_extent()
#print("Extent1", extent)

#fig.savefig('figure.png', bbox_inches=extent)


#Save 3D Landmarks Image

"""
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('3d_landmarks_alone.png', bbox_inches=extent)
"""

def get_features_from_image(image):
    """
    image_path: array 
        image object .

    """

   
    # Run the 3D face alignment on a test image, withCUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
    
    preds = fa.get_landmarks(image)[-1]

    

    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }
    return preds, pred_types




def image_2d_landmarks(image_path):
    """
    image_path: string
        path to image.

    """

    try:
        input_img = io.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        

    preds,pred_types = get_features_from_image(input_img )



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

    for pred_type in pred_types.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    fig.savefig('2d_landmarks.png')








def image_3d_landmarks():
    pass


if __name__ == "__main__":
    filepath = 'john_wick.jpg'
    image_2d_landmarks(filepath)