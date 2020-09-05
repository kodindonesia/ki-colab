# Colab helper that needs to install additional sw into Colab 
# https://colab.research.google.com/github/idealo/image-super-resolution/blob/master/notebooks/ISR_Prediction_Tutorial.ipynb

import numpy as np
from PIL import Image as PIL_Image
import cv2 as cv
import subprocess # to install additional libraries
from google.colab.patches import cv2_imshow  
import matplotlib.pyplot as plt
import io
import IPython.display

############################################################
class Pil_image:
  def get(image_address):
    return PIL_Image.open(image_address, 'r')
  def show(image):
    a = np.uint8(image)
    f = io.BytesIO()
    PIL_Image.fromarray(a).save(f, 'png')
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

############################################################
class Cv_image:
  def get(image_address):
    return cv.imread(image_address)
  def show(image):
    cv2_imshow(image)

############################################################
class Image_Super_Resolution: # does not need the GPU
  def __init__(self):
    self.model = None
    try:
      from ISR.models import RRDN
    except:
      result = subprocess.call(['pip', 'install', 'ISR']) 
    try:
      from ISR.models import RRDN
      self.model = RRDN(weights='gans')
    except:
      print("Image_Super_Resolution: SETUP FAILED!")

  @staticmethod
  def __get_PIL_image(image):
    if hasattr(image, "im"): # it is a PIL image
      return image
    else:
      return PIL_Image.fromarray(image)

  @staticmethod
  def __get_array_image(image):
    if hasattr(image, "im"): # it is a PIL image
      return np.array(image)
    else: # I assume that it is a np array
      return image

  def super_upscale_PIL_image_by_4x(self, image):
    img_array = self.__get_array_image(image)
    return PIL_Image.fromarray( self.model.predict(img_array) )

  def super_upscale_numpyimage_by_4x(self, image):
    img_array = self.__get_array_image(image)
    return self.model.predict(img_array) 

  def super_upscale_cvimage_by_4x(self, image):
    return self.super_upscale_numpyimage_by_4x(image)      

  def simple_upscale_PIL_image(self, image, multiplier=4):
    m = int(multiplier)
    if multiplier != m:
      print("simple_upscale_PIL_image: multiplier must be an integer! using ", m)
    pimg = self.__get_PIL_image(image)  
    return pimg.resize(size=(pimg.size[0]*m, pimg.size[1]*m), resample=PIL_Image.LANCZOS) 

  def simple_upscale_cvimage_image(self, image, multiplier=4):
    width = int(image.shape[1] * multiplier)
    height = int(image.shape[0] * multiplier)
    dim = (width, height)
    return cv.resize(image, dim, interpolation = cv.INTER_LANCZOS4)   


############################################################
class Image_Noise_Cancel: # does not need the GPU
  def __init__(self):
    self.model = None
    try:
      from ISR.models import RDN
    except:
      result = subprocess.call(['pip', 'install', 'ISR']) 
    try:
      from ISR.models import RDN
      self.model = RDN(weights='noise-cancel')
    except:
      print("Image_Noise_Cancel: SETUP FAILED!")

  @staticmethod
  def __get_array_image(image):
    if hasattr(image, "im"): # it is a PIL image
      return np.array(image)
    else: # I assume that it is a np array
      return image

  # PIL
  def clean_PIL_image_and_scale_by_2x(self, image):
    img_array = self.__get_array_image(image)
    return PIL_Image.fromarray( self.model.predict(img_array) )

  # open_CV

  def clean__numpyimage_and_scale_by_2x(self, image):
    img_array = self.__get_array_image(image)
    return self.model.predict(img_array)

  def clean_cvimage_and_scale_by_2x(self, image):
    return self.clean__numpyimage_and_scale_by_2x(image)

