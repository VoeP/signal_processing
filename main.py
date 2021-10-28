"""
This programm cuts the image at the second highly illuminated layer (= photoreceptor layer)
by searching for all edges and then sets everything aboth the second layer to 0.

variable parameters : sigma (for edges), sigma (for gauss.), guard

can be done on individual images
or

automated:
1. copy folders of SRF / NoSRF into dir
    images are converted by 56+...(one round dir= "SRF", ..)
2. add two folders named "outputs SRF" and "outputs NoSRF" -> cut files will be stored there

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import os
from scipy import ndimage
from skimage import feature
from PIL import Image
import straightline
import cv2
import matplotlib.pyplot as plt
import functions
plt.rcParams['image.cmap'] = 'gray'
import pandas as pd



"""
#convert and save to JPG
dir = r'Test'
for filename in os.listdir(dir):
    if filename.endswith(".png"):
        imgfile = dir + "/" + filename
        im = Image.open(imgfile)
        im = im.convert('RGB')
        newfile = filename.split(".")
        im.save(dir + "/" + newfile[0] + ".jpg")
print("all files converted in " + dir)
"""

#read image, extract first channel, filter and cut (individual)
"""
image1 = plt.imread('SRF/input_1504_1.jpg')
imagecopy = np.copy(image1[15:525, 47:557, 0])

edges = functions.edgemap(imagecopy)
result = functions.segmentLayer(imagecopy, edges, "output_1.png")

"""


"""
automated save of outputs
"""

distances = []
el = 0

dir = r'Test'
for filename in os.listdir(dir):
    if filename.endswith(".png"):
        imagefile = dir + "/" + filename

        imagefile_out_1 = "outputs " + dir + "/" + filename.split(".")[0] + ".png"
        imagefile_out_2 = "line " + dir + "/" + filename.split(".")[0] + ".png"
        image1 = plt.imread(imagefile)
        imagecopy = np.copy(image1[15:525, 47:557, 0]) # cut image at white borders

        #segment one layer
        edges = functions.edgemap(imagecopy)
        result = functions.segmentLayer(imagecopy, edges, imagefile_out_1)

        """
        Function for comparing the layers to a ransac
        """
        """
        #compare one layer to fitted line
        image = result[0]
        line = result[1]
        line_param = straightline.ransac(image, imagefile_out_2)
        distances.append(int(straightline.calculate_dist(line_param, line))),# imagefile))
        el +=1
        """

print("all images cut and saved in outputs " + dir)


"""
Second Part: Blob detection and Image categorization
"""
params = cv2.SimpleBlobDetector_Params()

##Area filtering parameter, refers to the blob size
params.filterByArea = True
params.minArea = 200

##Inertia basically refers to the similarity to a circle, 1 being a circle and 0 being a line
params.filterByInertia = True
params.minInertiaRatio = 0.05

##Filtering blobs by color, a treshold is defined, filterByColor=0 means that it's looking for black blobs
params.filterByColor = True
params.blobColor = 0
params.minThreshold = 0
params.maxThreshold = 60

##Circularity and convexity filtering:
params.filterByCircularity = True
params.minCircularity = 0.2
params.filterByConvexity = True
params.minConvexity = 0.2

a = functions.test_for_srf("Test", "outputs Test", params)
#for i in range(0, len(a[0])):
#    print(a[0][i], a[1][i])

df = pd.DataFrame(data={"filename": a[0], "label": a[1]})
df.to_csv("./bernhardsgruetter_cristea_paukku.csv", sep=',', index=False)



