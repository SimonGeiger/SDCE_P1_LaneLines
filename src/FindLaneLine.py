import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Line():
  """
  docstring
  """
  def __init__(self, *args):
    if len(args) == 4:
      self.start_x = args[0]
      self.start_y = args[1]
      self.end_x = args[2]
      self.end_y = args[3]

      if self.end_x != self.start_x:
        self.m = (self.end_y - self.start_y) / (self.end_x - self.start_x)
        self.b = self.start_y - self.m * self.start_x 
      else:
        self.m = np.Infinity
        self.b = np.Infinity

      # self.length = np.sqrt( (self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)

      m_wanted_left = 0.55
      m_wanted_right = -0.68
      m_tolerance = 0.25
      if self.m > (m_wanted_left - m_tolerance) and  self.m < (m_wanted_left + m_tolerance):
          # left line => / 
          self.type = 'left'
      elif self.m > (m_wanted_right - m_tolerance) and self.m < (m_wanted_right + m_tolerance):
          # right line => \ 
          self.type = 'right'
      else:
          # line to the left or to the right => not useful
          self.type = 'horizontal'
    elif len(args) == 3:
      self.m = args[0]
      self.b = args[1]
      self.start_y = 540
      self.start_x = int((self.start_y - self.b) / self.m)
      self.end_y = 330
      self.end_x = int((self.end_y - self.b) / self.m)
      self.type = args[2]

    else:
      raise NotImplementedError(f"Constructor for {len(args)} has not been defined")

  def draw_line(self, img, color = [255, 0, 0], thickness = 2):
    cv2.line(img, (self.start_x, self.start_y), (self.end_x, self.end_y), color, thickness)  


class Image:
  """
  docstring
  """

  def __init__(self, path_to_image):
    # set font size of plot figure to 10 pt
    plt.rcParams.update({'font.size': 10})

    # plot figure in max window size
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

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
    
    ### blur image to get rid of artifacts => isn't helpful for overall result
    
    kernel_size = 5
    self.image_hsv = cv2.GaussianBlur(self.image_hsv, (kernel_size, kernel_size), 0)
    # self.mask_yellow_color = cv2.GaussianBlur(self.mask_yellow_color, (kernel_size, kernel_size), 0)
    # self.mask_white_color = cv2.GaussianBlur(self.mask_white_color, (kernel_size, kernel_size), 0)

    ### create color mask by filtering for yellow and white 
    lower_yellow = np.array([17, 60, 0], dtype="uint8")
    upper_yellow = np.array([25, 220, 255], dtype="uint8")
    self.mask_yellow_color = cv2.inRange(self.image_hsv, lower_yellow, upper_yellow)
    lower_white = np.array([0,0,130], dtype="uint8")
    upper_white = np.array([255,60,255], dtype="uint8")
    self.mask_white_color = cv2.inRange(self.image_hsv, lower_white, upper_white)

  
    ### combine color masks and then mask image
    self.color_mask = cv2.bitwise_or(self.mask_white_color,self.mask_yellow_color)
    # self.color_mask = cv2.threshold(self.color_mask, 127, 255, cv2.THRESH_BINARY)[1]
    self.masked_image = cv2.bitwise_and(self.image_raw, np.repeat(self.color_mask[:, :, np.newaxis], 3, axis=2))


  def find_hough_lines_in_roi(self):
    if(self.color_mask.all() == None):
      raise ValueError(f"Mask is missing. Please run 'create_color_mask' first.")

    ### mask with region of interest
    vertices = np.array([[[80,540], [920,540], [515,320], [445,320]]], dtype=np.int32)      # both lanes
    # vertices = np.array([[[80,540], [500,540], [500,320], [445,320]]], dtype=np.int32)      # only left lane
    # vertices = np.array([[[500,540], [920,540], [515,320], [500,320]]], dtype=np.int32)     # only right lane

    self.roi_mask = np.zeros_like(self.color_mask)
    cv2.fillPoly(self.roi_mask, vertices, 255)
    self.color_mask_roi = cv2.bitwise_and(self.color_mask, self.roi_mask)    

    ### apply canny edge detection
    self.canny = cv2.Canny(self.color_mask_roi, threshold1=500, threshold2=1500)

    ### apply hough transformation and find long edges
    self.hough_lines = cv2.HoughLinesP(self.canny, rho = 1, theta = np.pi/180, threshold = 30, lines=np.array([]), minLineLength = 20, maxLineGap = 200)

    ### draw found lines into image
    self.hough_image = np.repeat(self.canny[:, :, np.newaxis], 3, axis=2)
    self.canny_hough_image = np.repeat(self.canny[:, :, np.newaxis], 3, axis=2)

    self.lines = list()
    for i, element in enumerate(self.hough_lines):
      self.lines.append(Line(element[0][0],element[0][1],element[0][2],element[0][3]))
      self.lines[i].draw_line(self.hough_image, thickness = 1)
      self.lines[i].draw_line(self.canny_hough_image, thickness = 1)
    # self.hough_image = cv2.addWeighted(src1 = self.image_raw, alpha = 0.8, src2 = self.hough_image, beta = 1.0, gamma = 0.0)

  def get_lanes(self):
    list_left = list()
    list_right = list()
    for element in self.lines:
      if element.type == 'left':
        list_left.append([element.m, element.b])
      elif element.type == 'right':
        list_right.append([element.m, element.b])

    self.lane_left = [np.mean([item[0] for item in list_left]),np.mean([item[1] for item in list_left])]
    self.lane_right = [np.mean([item[0] for item in list_right]),np.mean([item[1] for item in list_right])]

    left = Line(*self.lane_left, 'left')
    right = Line(*self.lane_right, 'right')
    self.lane_image = np.zeros_like(self.color_mask) 
    self.lane_image = np.repeat(self.lane_image[:, :, np.newaxis], 3, axis=2)
    left.draw_line(self.lane_image, thickness = 8)
    right.draw_line(self.lane_image, thickness = 8)   
    self.lane_image = cv2.addWeighted(src1 = self.image_raw, alpha = 0.8, src2 = self.lane_image, beta = 1.0, gamma = 0.0)

    return [self.lane_left, self.lane_right]


  def print(self, mode):
    if mode == 0: # show raw color image with shape info
      print(f"The shape of the image is {self.shape}.")
      plt.title("RGB image")
      plt.imshow(self.image_raw)

    elif mode == 0.5: # compare HSV and HSL
      plt.subplot(1,3,1)
      plt.title("RGB color space")
      plt.imshow(self.image_raw)
      plt.subplot(1,3,2)
      plt.title("HSV color space")
      plt.imshow(cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HSV))
      plt.subplot(1,3,3)
      plt.title("HLS color space")
      plt.imshow(cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HLS))
      
    elif mode == 1: # show raw image, color mask + masked result
      plt.subplot(2,2,1)
      plt.title("rgb image")
      plt.imshow(self.image_raw)
      plt.subplot(2,2,2)
      plt.title("combined color mask")
      plt.imshow(self.color_mask, cmap='gray')
      # plt.imshow(self.masked_image, cmap='gray')
      plt.subplot(2,2,3)
      plt.title("yellow color mask")
      plt.imshow(self.mask_yellow_color, cmap='gray')
      plt.subplot(2,2,4)
      plt.title("white color mask")
      plt.imshow(self.mask_white_color, cmap='gray')

    elif mode == 2: # show result of edge detection and hough transformation
      plt.subplot(2,2,1)
      plt.title("color mask")
      plt.imshow(self.color_mask, cmap='gray')
      plt.subplot(2,2,2)
      plt.title("canny edge detection with hough transformation")
      plt.imshow(self.canny_hough_image)
      plt.subplot(2,2,3)
      plt.title("color mask with ROI only")
      plt.imshow(self.color_mask_roi, cmap = 'gray')
      plt.subplot(2,2,4)
      plt.title("canny edge detection")
      plt.imshow(self.canny, cmap='gray')

    elif mode == 3:
      plt.subplot(2,2,1)
      plt.title("RGB image")
      plt.imshow(self.image_raw)
      plt.subplot(2,2,2)
      plt.title("image with lanes marked")
      plt.imshow(self.lane_image)
      plt.subplot(2,2,3)
      plt.title("color mask")
      plt.imshow(self.color_mask, cmap='gray')
      plt.subplot(2,2,4)
      plt.title("canny edge detection with hough transformation")
      plt.imshow(self.canny_hough_image)

    plt.show()