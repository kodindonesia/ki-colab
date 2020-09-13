# https://towardsdatascience.com/image-read-and-resize-with-opencv-tensorflow-and-pil-3e0f29b992be

# super resolution:
# https://www.tensorflow.org/hub/tutorials/image_enhancing
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_enhancing.ipynb#scrollTo=IslbQmTj0ukz
# https://tfhub.dev/

# numpy image
import numpy as np
# opencv
import cv2 as cv
from google.colab.patches import cv2_imshow  
# PIL
from PIL import Image as PIL_Image
import io
import IPython.display
# tfimage
import tensorflow as tf # tested with tf 2

############################################################
class Pil_image:
  def get(image_address):
    return PIL_Image.open(image_address, 'r')

  def write(pil_image, file_path):
    pil_image.save(file_path)  

  def show(pil_image):
    a = np.uint8(pil_image)
    f = io.BytesIO()
    PIL_Image.fromarray(a).save(f, 'png')
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

  def resize(pil_image, multiplier):
    size=(int(pil_image.size[0]*multiplier+0.5), int(pil_image.size[1]*multiplier+0.5))
    return pil_image.resize(size=size, resample=PIL_Image.LANCZOS)     

############################################################
class Cv_image:
  def get(image_address, no_alpha=False, gray=False):
    flag = cv.IMREAD_UNCHANGED
    if gray:
      flag = cv.IMREAD_GRAYSCALE
    elif no_alpha:
      flag = cv.IMREAD_COLOR
    return cv.imread(image_address, flag) # row (height) x column (width) x color (3). The order of color is BGR (blue, green, red).

  def write(cv_image, file_path):
    cv.imwrite(file_path, cv_image) 

  def show(cv_image):
    cv2_imshow(cv_image)

  def resize(cv_image, multiplier):
    width = int(cv_image.shape[1] * multiplier + 0.5)
    height = int(cv_image.shape[0] * multiplier + 0.5)
    size = (width, height)
    return cv.resize(cv_image, size, interpolation=cv.INTER_LANCZOS4)   

  def to_numpyrgb_image(cv_image):
    return cv.cvtColor(cv_image, cv.COLOR_BGR2RGB) # row (height) x column (width) x color . The order of color is RGB.


############################################################
class Tf_image:
  def get(image_address, no_alpha=False, gray=False):
    if gray:
      return tf.image.decode_image(tf.io.read_file(image_address), channels=1, expand_animations=False) # BMP, GIF, JPEG, or PNG
    tf_image = tf.image.decode_image(tf.io.read_file(image_address), expand_animations=False) # BMP, GIF, JPEG, or PNG
    if no_alpha:
      # If PNG, remove the alpha channel. The model only supports
      # images with 3 color channels.
      if tf_image.shape[-1] == 4:
        tf_image = tf_image[...,:-1]      
    return tf_image  

  def to_numpyrgb_image(tf_image):
    image = tf.clip_by_value(tf_image, 0, 255)
    return tf.cast(image, tf.uint8).numpy() # row (height) x column (width) x color . The order of color is RGB.

  def to_cv_image(tf_image):
    return cv.cvtColor(Tf_image.to_numpyrgb_image(tf_image), cv.COLOR_BGR2RGB) # row (height) x column (width) x color . The order of color is BGR.

  def show(tf_image):
    cv2_imshow( Tf_image.to_cv_image(tf_image) )

  def resize(tf_image, multiplier):
    # resizing the image with tensorflow 2.x
    #print("image shape before resize:", tf_image.shape)
    #print("image dtype: ", tf_image.dtype)
    #This function takes in a 4D input. Hence we will have to expand the image and then squeeze back to three dimensions before we can use it as an image.
    image_tf_4D= tf.expand_dims(tf_image,0)
    # doing bilinear resize
    width = int(tf_image.shape[1] * multiplier + 0.5)
    height = int(tf_image.shape[0] * multiplier + 0.5)
    size = (width, height)
    image_tf_resized_4D = tf.image.resize( image_tf_4D, size )
    #squeezing back the image to 3D
    image_tf_resized = tf.squeeze(image_tf_resized_4D)
    #Above is still a tensor. So we need to convert it to numpy. We do this by using tf session.
    image_tf_resized = tf.cast(image_tf_resized, tf.uint8)
    #print("image shape after resize:", image_tf_resized.shape)
    #print("image dtype: ", image_tf_resized.dtype)
    return image_tf_resized



#########################################################################################################
######################################################################### esrgan
# https://www.tensorflow.org/hub/tutorials/image_enhancing
# source: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_enhancing.ipynb
class Image_Super_Resolution: # does not need the GPU
  def __init__(self):
    import os
    import tensorflow as tf
    import tensorflow_hub as hub
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';  logging.getLogger("tensorflow").setLevel(logging.ERROR) # should stop info and warnings
    os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "False";  SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    self.model = hub.load(SAVED_MODEL_PATH)

  @staticmethod
  def __get_preprocessed_image_from_tfimage(hr_image):
    if hr_image.shape[-1] == 4: # If PNG, remove the alpha channel.
      hr_image = hr_image[...,:-1] # The model only supports images with 3 color channels.
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32) # The image must be a float32 image, converted using tf.cast(image, tf.float32).
    return tf.expand_dims(hr_image, 0) # The image must be of 4 dimensions, [batch_size, height, width, 3]. To perform super resolution on a single image, use tf.expand_dims(image, 0) to add the batch dimension.

  @staticmethod
  def __get_preprocessed_image_from_path(image_path):
    """ Loads image from path and preprocesses to make it model ready
        Args:
          image_path: Path to the image file
    """
    tf_image = tf.image.decode_image(tf.io.read_file(image_path))
    return Image_Super_Resolution.__get_preprocessed_image_from_tfimage(tf_image)


  def __get_tfimage_upscaled4x(self, preprocessed_image):
    fake_image = self.model(preprocessed_image) #the model output has got 4 dimensions, [batch_size, height, width, 3]
    fake_image = tf.squeeze(fake_image)# we get a tf tensor with 3 dimensions [height, width, 3]
    image = np.asarray(fake_image) # we move the GPU tensor into CPU memory numpy array  [height, width, 3]
    image = tf.clip_by_value(image, 0, 255) # we ensure the floating point array is in the range 0-255 for r and g and b
    return tf.cast(image, tf.uint8) # from floating point32 to integer rgb

  def __get_numpyrgbimage_upscaled4x(self, preprocessed_image):
    return self.__get_tfimage_upscaled4x(preprocessed_image).numpy() # from tensor integer rgb to numpy integer rgb 

  def __get_pilimage_upscaled4x_from_preprocessed_image(self, preprocessed_image):
    numpy_rgb = self.__get_numpyrgbimage_upscaled4x(preprocessed_image)
    return PIL_Image.fromarray( numpy_rgb ) 

  def __get_cvimage_upscaled4x_from_preprocessed_image(self, preprocessed_image):
    numpy_rgb = self.__get_numpyrgbimage_upscaled4x(preprocessed_image)
    return cv.cvtColor(numpy_rgb, cv.COLOR_BGR2RGB) # convert to BGR open_cv image

  # API from file path

  def get_pilimage_upscaled4x_from_file_path(self, image_path):
    preprocessed_image = self.__get_preprocessed_image_from_path(image_path)
    return self.__get_pilimage_upscaled4x_from_preprocessed_image(preprocessed_image)

  def get_cvimage_upscaled4x_from_file_path(self, image_path):
    preprocessed_image = self.__get_preprocessed_image_from_path(image_path)
    return self.__get_cvimage_upscaled4x_from_preprocessed_image(preprocessed_image) 

  def get_numpyrgbimage_upscaled4x_from_file_path(self, image_path):
    preprocessed_image = self.__get_preprocessed_image_from_path(image_path)
    return self.__get_numpyrgbimage_upscaled4x(preprocessed_image)

  def get_tfimage_upscaled4x_from_file_path(self, image_path):
    preprocessed_image = self.__get_preprocessed_image_from_path(image_path)
    return self.__get_tfimage_upscaled4x(preprocessed_image)

  # API from image

  def get_tfimage_upscaled4x(self, tf_image):
    preprocessed_image = self.__get_preprocessed_image_from_tfimage(tf_image)
    return self.__get_tfimage_upscaled4x(preprocessed_image)

  def get_numpyrgbimage_upscaled4x(self, numpyrgb_image):
    preprocessed_image = self.__get_preprocessed_image_from_tfimage(numpyrgb_image)
    return self.__get_numpyrgbimage_upscaled4x(preprocessed_image)

  def get_pilimage_upscaled4x(self, pil_image):
    numpyrgb_image = np.array(pil_image)
    preprocessed_image = self.__get_preprocessed_image_from_tfimage(numpyrgb_image)
    return self.__get_pilimage_upscaled4x_from_preprocessed_image(preprocessed_image)
   
  def get_cvimage_upscaled4x(self, cv_image):
    return self.get_numpyrgbimage_upscaled4x(cv_image)


# test
# tf image
#from kicolab.image_helper import Tf_image, Image_Super_Resolution 
#
#isr = Image_Super_Resolution()
#
#tfimage = Tf_image.get(IMAGE_PATH)
#print(tfimage.shape, type(tfimage))
#Tf_image.show(tfimage)
#
#tfimage4x = Tf_image.resize(tfimage, 4)
#print(tfimage4x.shape, type(tfimage4x))
#Tf_image.show(tfimage4x)
#
#tfimage_super =isr.get_tfimage_upscaled4x(tfimage)
#print(tfimage_super.shape, type(tfimage_super))
#Tf_image.show(tfimage_super)
#
#tfimage_super = isr.get_tfimage_upscaled4x_from_file_path(IMAGE_PATH)
#print(tfimage_super.shape, type(tfimage_super))
#Tf_image.show(tfimage_super)    
