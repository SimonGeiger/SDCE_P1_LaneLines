import numpy as np
import cv2

class Line():
  """
  This Line class stores the parameters (start and end points, slope m and y-interception b) of every line.
  Additionally it evaluates (depending on slope value) if the line is potentially part of the left or right lane
  """
  def __init__(self, *args):
    '''
    This constructor initializes the line object. Two cases are accepted as input [start + end point + image_shape]
    or [slope + y-interception + classication + image_shape]. Lines are drawn depending on image size.
    '''
    ### constructor for start and end point input
    if len(args) == 5:
      self.start_x = args[0]
      self.start_y = args[1]
      self.end_x = args[2]
      self.end_y = args[3]
      self.image_shape = args[4]

      if self.end_x != self.start_x:
        self.m = (self.end_y - self.start_y) / (self.end_x - self.start_x)
        self.b = self.start_y - self.m * self.start_x 
      else:
        self.m = np.Infinity
        self.b = np.Infinity

      # self.length = np.sqrt( (self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)

      ### orientation classification depending on value of slope m
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

    ### constructor for slope, y-interception and classifiation as input
    elif len(args) == 4:
      self.m = args[0]
      self.b = args[1]
      self.type = args[2]
      self.image_shape = args[3]
      self.start_y = np.int32(np.rint(self.image_shape[1]/960 * 540)) # lower edge of image
      self.start_x = np.int32(np.rint((self.start_y - self.b) / self.m))
      self.end_y = np.int32(np.rint(self.image_shape[1]/960 * 330)) # draw line up to this height
      self.end_x = np.int32(np.rint((self.end_y - self.b) / self.m))

    ### error, because wrong number of inputs
    else:
      raise NotImplementedError(f"Constructor for {len(args)} has not been defined")

  def draw_line(self, img, color = [255, 0, 255], thickness = 2):
    '''
    This function draws a line into a given image. Color and thickness can be parametrizied.
    '''
    cv2.line(img, (self.start_x, self.start_y), (self.end_x, self.end_y), color, thickness)  

