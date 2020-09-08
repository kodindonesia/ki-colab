# Colab helper that needs to install additional sw into Colab (no need for GPU but can run faster with it)
# original Deep Learning model by Google, Apache License 2
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/style_transfer/overview.ipynb
# part of the series: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/g3doc/models

import tensorflow as tf # tested with TF 2
import IPython.display as display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
#
import cv2 as cv

############################################################
class Style_transfer: # does not need the GPU but fastest model can run faster with a GPU
  def __init__(self, style_path=None, fastest=True, verbose=False):
    self.verbose = verbose
    self.fastest = fastest
    if fastest:
      self.style_predict_path = tf.keras.utils.get_file('style_predict.tflite.int8', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
      self.style_transform_path = tf.keras.utils.get_file('style_transform.tflite.int8', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
    else:
      self.style_predict_path = tf.keras.utils.get_file('style_predict.tflite.fp16', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/prediction/1?lite-format=tflite')
      self.style_transform_path = tf.keras.utils.get_file('style_transform.tflite.fp16', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/transfer/1?lite-format=tflite')
    self.set_style(style_path)
    self.preprocessed_content_image = None
    self.unprocessed_content_image = None  # useful when blending style and content images
    self.stylized_image = None
    self.style_bottleneck_content = None # useful when blending style and content images



  @staticmethod
  # Function to load an image from a file, and add a batch dimension.
  def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

  @staticmethod
  # Function to pre-process by resizing an central cropping it.
  def __preprocess_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image

  @staticmethod
  def set_matplotlib(figures_num, rows=1):
    mpl.rcParams['figure.figsize'] = (8 * figures_num, 8 * rows)
    mpl.rcParams['axes.grid'] = False

  @staticmethod
  def __show_preprocessed_image(image, title=None):
    if len(image.shape) > 3:
      image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
      plt.title(title)


  def set_style(self, path_to_img):
    if path_to_img is not None and path_to_img[-4:].lower() != '.jpg' and path_to_img[-5:].lower() != '.jpeg':
      print("Style_transfer.set_style: ERROR: for the moment, only .jpg files are accepted")
      path_to_img = None
    file_location = 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg' if path_to_img is None else path_to_img
    file_base_name = os.path.basename(file_location)    
    file_path = tf.keras.utils.get_file(file_base_name, file_location)
    image = self.load_img(file_path) # Load the input image.
    self.preprocessed_style_image = self.__preprocess_image(image, 256)# Preprocess the style image.
    self.style_bottleneck = self.__run_style_predict(self.preprocessed_style_image)# Calculate style bottleneck for the preprocessed style image.
    if self.verbose:
      self.set_matplotlib(1)
      self.__show_preprocessed_image(self.preprocessed_style_image, 'Style Image')

  def set_content(self, path_to_img):
    if path_to_img is not None and path_to_img[-4:].lower() != '.jpg':
      print("Style_transfer.set_content: ERROR: for the moment, only .jpg files are accepted")
      return
    file_base_name = os.path.basename(path_to_img)
    file_path = tf.keras.utils.get_file(file_base_name, path_to_img)  
    self.unprocessed_content_image = self.load_img(file_path) # Load the input image.
    self.preprocessed_content_image = self.__preprocess_image(self.unprocessed_content_image, 384) # Preprocess the image.
    self.style_bottleneck_content = None
    if self.verbose:
      self.set_matplotlib(1)
      self.__show_preprocessed_image(self.preprocessed_content_image, 'Content Image')

  # Function to run style prediction on preprocessed style image.
  def __run_style_predict(self, preprocessed_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=self.style_predict_path)
    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_image)
    # Calculate style bottleneck.
    interpreter.invoke()
    return interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
        )()


  # Stylize the content image using the style bottleneck.
  def __run_style_transform(self, style_bottleneck):
    if self.preprocessed_content_image is None:
      print("Style_transfer: ERROR before this you need: set_content(path_to_img)")
      return
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=self.style_transform_path)
    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], self.preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()
    # Transform content image.
    return interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
        )()

  # Run style transform on preprocessed style image
  def create_stylized_image(self):
    # Transform content image.
    self.stylized_image = self.__run_style_transform(self.style_bottleneck)
    if self.verbose:    
      #print('Stylized Image Shape:', self.stylized_image.shape)
      self.set_matplotlib(1)
      self.__show_preprocessed_image(self.stylized_image, 'Stylized Image')


  def create_stylized_and_content_image_blend(self, blend=0.5):
    if blend<=0 or blend>=1:
      print("Style_transfer: ERROR blend must be between 0 and 1")
      return
    if self.style_bottleneck_content is None:
      # Calculate style bottleneck of the content image.
      self.style_bottleneck_content = self.__run_style_predict( self.__preprocess_image(self.unprocessed_content_image, 256) )  
    # Blend the style bottleneck of style image and content image
    style_bottleneck_blended = blend * self.style_bottleneck_content \
                              + (1 - blend) * self.style_bottleneck
    # Stylize the content image using the style bottleneck.
    self.stylized_image = self.__run_style_transform(style_bottleneck_blended)
    if self.verbose:    
      #print('Stylized Image Shape:', self.stylized_image.shape)
      self.set_matplotlib(1)
      self.__show_preprocessed_image(self.stylized_image, 'Stylized Image (blended ' + str(blend) + ')' )

  def show_content_and_style(self):
    self.set_matplotlib(2)
    if self.preprocessed_content_image is not None:
      plt.subplot(1, 2, 1)
      self.__show_preprocessed_image(self.preprocessed_content_image, 'Content Image')
    plt.subplot(1, 2, 2)
    self.__show_preprocessed_image(self.preprocessed_style_image, 'Style Image')

  def show_stylized_and_content_and_style(self):
    self.set_matplotlib(3)
    if self.stylized_image is not None:
      plt.subplot(1, 3, 1)
      self.__show_preprocessed_image(self.stylized_image, 'Stylized Image')
    if self.preprocessed_content_image is not None:
      plt.subplot(1, 3, 2)
      self.__show_preprocessed_image(self.preprocessed_content_image, 'Content Image')
    plt.subplot(1, 3, 3)
    self.__show_preprocessed_image(self.preprocessed_style_image, 'Style Image')

  # open_CV #######################

  def get_opencv_stylized(self):
    return cv.cvtColor(np.uint8(tf.squeeze(self.stylized_image*255.9999, axis=0).numpy()), cv.COLOR_RGB2BGR)

  def get_opencv_content_original(self):
    return np.uint8(cv.cvtColor(tf.squeeze(self.unprocessed_content_image*255.9999, axis=0).numpy(), cv.COLOR_RGB2BGR))

  def get_opencv_content(self):
    return np.uint8(cv.cvtColor(tf.squeeze(self.preprocessed_content_image*255.9999, axis=0).numpy(), cv.COLOR_RGB2BGR))
