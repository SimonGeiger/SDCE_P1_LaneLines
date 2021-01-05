import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class MyImage:
  """
  docstring
  """

  def __init__(self, path_to_image):
    # set font size of plot figure to 10 pt
    plt.rcParams.update({'font.size': 10})

    # read image and convert to RGB colors
    if path_to_image != None:
      self.image_raw = cv2.cvtColor(cv2.imread(path_to_image,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
      self.shape = self.image_raw.shape
    else:
      # introduction to python exceptions => https://realpython.com/python-exceptions/
      # how/when to use assert, raise, try/except => https://stackoverflow.com/questions/40182944/difference-between-raise-try-and-assert
      raise ValueError(f"Path to Image is missing.")


  def create_color_mask(self):
    ### convert to HSV color space => https://en.wikipedia.org/wiki/HSL_and_HSV#Basic_principle
    self.image_hsv = cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HSV)
    self.image_hls = cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HLS)
    ### create color mask by filtering for yellow and white 
    lower_yellow = np.array([17, 60, 0], dtype="uint8")
    upper_yellow = np.array([25, 220, 255], dtype="uint8")
    self.mask_yellow_color = cv2.inRange(self.image_hsv, lower_yellow, upper_yellow)
    lower_white = np.array([0,0,130], dtype="uint8")
    upper_white = np.array([255,60,255], dtype="uint8")
    self.mask_white_color = cv2.inRange(self.image_hsv, lower_white, upper_white)

    ### blur image to get rid of artifacts => doesn't seem helpful for overall result
    # kernel_size = 5
    # self.mask_yellow_color = cv2.GaussianBlur(self.mask_yellow_color, (kernel_size, kernel_size), 0)
    # self.mask_white_color = cv2.GaussianBlur(self.mask_white_color, (kernel_size, kernel_size), 0)

    ### combine color masks and then mask image
    self.color_mask = cv2.bitwise_or(self.mask_white_color,self.mask_yellow_color)
    self.masked_image = cv2.bitwise_and(self.image_raw, np.repeat(self.color_mask[:, :, np.newaxis], 3, axis=2))


  def find_edges(self):
    if(self.color_mask.all() == None):
      raise ValueError(f"Mask is missing. Please run 'create_color_mask' first.")

    lower_canny = 500
    upper_canny = 1500
    self.canny = cv2.Canny(self.color_mask, lower_canny, upper_canny)

    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_len = 8
    max_line_gap = 5
    self.hough = cv2.HoughLinesP(self.canny, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # TODO: Draw Lines into image


  def print(self, mode):
    if mode == 0: # show raw color image with shape info
      print(f"The shape of the image is {self.shape}.")
      plt.title("raw color image")
      plt.imshow(self.image_raw)

    if mode == 0.5: # compare HSV and HSL
      plt.subplot(1,3,1)
      plt.title("raw color image")
      plt.imshow(self.image_raw)
      plt.subplot(1,3,2)
      plt.title("hsv image")
      plt.imshow(self.image_hsv)
      plt.subplot(1,3,3)
      plt.title("hls image")
      plt.imshow(self.image_hls)
      
    elif mode == 1: # show raw image, color mask + masked result
      plt.subplot(2,2,1)
      plt.title("rgb image")
      plt.imshow(self.image_raw)
      plt.subplot(2,2,2)
      plt.title("color masked image")
      plt.imshow(self.color_mask, cmap='gray')
      # plt.imshow(self.masked_image, cmap='gray')
      plt.subplot(2,2,3)
      plt.title("yellow color mask")
      plt.imshow(self.mask_yellow_color, cmap='gray')
      plt.subplot(2,2,4)
      plt.title("white color mask")
      plt.imshow(self.mask_white_color, cmap='gray')

    elif mode ==2: # show result of edge detection and hough transformation
      plt.subplot(2,2,1)
      plt.title("color mask")
      plt.imshow(self.color_mask, cmap='gray')
      plt.subplot(2,2,2)
      plt.title("detected hough lines")
      # plt.imshow(self.color_mask, cmap='gray')
      plt.subplot(2,2,3)
      plt.title("canny edge detection")
      plt.imshow(self.canny, cmap='gray')
      plt.subplot(2,2,4)
      plt.title("hough transformation")
      # plt.imshow(self.hough, cmap='gray')
    plt.show()
