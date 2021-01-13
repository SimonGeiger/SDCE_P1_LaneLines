import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__" or __name__ == 'Image':
  from Line import Line
else:
  from src.Line import Line

class Image:
  """
  This class has all the functions implemented to process an image to detect the lanes.
  Call the process_image() function to find lane lines in the image.
    Steps:
    - filter for yellow and white colors as those represent the lanes
    - crop search area to the region of interest (ROI)
    - find edges with canny
    - find related edges with hough transformation
    - calculate slope and y-intercept for every hough line
    - classify line into left lane, right lane, horizontal via line slope
    - calculate mean slope/y-intercept from entire left lane lines and right lane lines
    - return image with lanes marked
  There's also a print function implemented to display certain outputs of the lane detection algorithm

  ToDo/Improvements:
    - split algorithm and visualization
    - catch potential edge cases like
      > bad input: no image
      > no visible lane markings
      > ...
    - improve algorithm speed
      > a lot of code artifacts only used for developing purposes
    - outsource parameters into header
    - stabilize output by using information/lane position from previous frames
    - improve to detect curves
    - hard coded parameters + ROI isn't flexible (for example driving up or downhill)
    - detection of adjacent lane
    - detection of road boundaries => better ROI?
  """

  def __init__(self):
    self.print_mode = 4

  def process_image(self, img, print_mode=0):
    """
    This function processes a single image and returns an image with detected lane lines highlighted.
    print_mode is an opptional value, with it steps of the algorithm can be visualized.
    """
    self.print_mode = print_mode

    self.readImage(img)
    self.create_color_mask()
    self.find_hough_lines_in_roi()
    self.get_lanes()

    if self.print_mode > 0:
      self.print(self.print_mode)

    return cv2.cvtColor(self.lane_image,cv2.COLOR_RGB2BGR)

  def readImage(self,img):
    # read image and convert to RGB colors
    if img.all() != None:
      self.image_raw = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      self.shape = self.image_raw.shape
    else:
    #   # introduction to python exceptions => https://realpython.com/python-exceptions/
    #   # how/when to use assert, raise, try/except => https://stackoverflow.com/questions/40182944/difference-between-raise-try-and-assert
      raise ValueError(f"Image is missing.")

  def create_color_mask(self):
    ### convert to HSV color space => https://en.wikipedia.org/wiki/HSL_and_HSV#Basic_principle
    self.image_hsv = cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HSV)
    
    ### blur image to get rid of artifacts => isn't helpful for overall result
    # kernel_size = 5
    # self.image_hsv = cv2.GaussianBlur(self.image_hsv, (kernel_size, kernel_size), 0)
    # self.mask_yellow_color = cv2.GaussianBlur(self.mask_yellow_color, (kernel_size, kernel_size), 0)
    # self.mask_white_color = cv2.GaussianBlur(self.mask_white_color, (kernel_size, kernel_size), 0)

    ### create color mask by filtering for yellow and white 
    if self.print_mode != 0:
      # range for test images in .png
      lower_yellow = np.array([17, 60, 0], dtype="uint8")
      upper_yellow = np.array([25, 220, 255], dtype="uint8")
    else:
      # range for clips in .mp4
      lower_yellow = np.array([75, 64, 0], dtype="uint8")
      upper_yellow = np.array([105, 255, 255], dtype="uint8")

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

    ### mask with region of interest, ROI is being extrapolated depending on image size, but optimized on 960x540 size
    vertices = np.int32(np.rint(self.shape[1]/960 * np.array([[[80,540], [920,540], [515,320], [445,320]]], dtype=np.int32))) 
    self.roi_mask = np.zeros_like(self.color_mask)
    cv2.fillPoly(self.roi_mask, vertices, 255)
    self.color_mask_roi = cv2.bitwise_and(self.color_mask, self.roi_mask)    

    ### apply canny edge detection
    self.canny = cv2.Canny(self.color_mask_roi, threshold1=500, threshold2=1500)

    ### apply hough transformation and find long edges
    self.hough_lines = cv2.HoughLinesP(self.canny, rho = 1, theta = np.pi/180, threshold = 30, lines=np.array([]), minLineLength = 20, maxLineGap = 100)

    ### draw found lines into image
    self.hough_image = np.repeat(self.canny[:, :, np.newaxis], 3, axis=2)
    self.canny_hough_image = np.repeat(self.canny[:, :, np.newaxis], 3, axis=2)

    self.lines = list()
    for i, element in enumerate(self.hough_lines):
      self.lines.append(Line(element[0][0],element[0][1],element[0][2],element[0][3], self.shape))
      # self.lines[i].draw_line(self.hough_image, color=[0,255,0], thickness = 1)
      self.lines[i].draw_line(self.canny_hough_image, thickness = 1)

  def get_lanes(self):
    ### get average lane parameters (slope/intersection) for left and right
    list_left = list()
    list_right = list()
    for element in self.lines:
      if element.type == 'left':
        list_left.append([element.m, element.b])
      elif element.type == 'right':
        list_right.append([element.m, element.b])
    
    self.lane_image = np.zeros_like(self.color_mask) 
    self.lane_image = np.repeat(self.lane_image[:, :, np.newaxis], 3, axis=2)

    if len(list_left) > 0:
      self.lane_left  = Line(np.mean([item[0] for item in list_left]),  np.mean([item[1] for item in list_left]), 'left', self.shape)
      self.lane_left.draw_line(self.lane_image, thickness = 8)
    if len(list_right) > 0:
      self.lane_right = Line(np.mean([item[0] for item in list_right]), np.mean([item[1] for item in list_right]), 'right', self.shape)
      self.lane_right.draw_line(self.lane_image, thickness = 8)   

    ### blend lane into original image
    self.lane_image = cv2.addWeighted(src1 = self.image_raw, alpha = 0.8, src2 = self.lane_image, beta = 1.0, gamma = 0.0)
    # self.lane_image = cv2.addWeighted(src1 = self.hough_image, alpha = 0.8, src2 = self.lane_image, beta = 1.0, gamma = 0.0)

  def print(self, mode):
    # set font size of plot figure to 10 pt
    plt.rcParams.update({'font.size': 10})

    # plot figure in max window size
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    if mode == 1: # show raw color image with shape info
      print(f"The shape of the image is {self.shape}.")
      plt.title("RGB image")
      plt.imshow(self.image_raw)

    elif mode == 1.5: # compare HSV and HSL
      plt.subplot(1,3,1)
      plt.title("RGB color space")
      plt.imshow(self.image_raw)
      plt.subplot(1,3,2)
      plt.title("HSV color space")
      plt.imshow(cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HSV))
      plt.subplot(1,3,3)
      plt.title("HLS color space")
      plt.imshow(cv2.cvtColor(self.image_raw, cv2.COLOR_RGB2HLS))
      
    elif mode == 2: # show raw image, color mask + masked result
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

    elif mode == 3: # show result of edge detection and hough transformation
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

    elif mode == 4: # show summary of all steps
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
      plt.title("canny edge detection with hough transformation within ROI")
      plt.imshow(self.canny_hough_image)

    plt.show()

    