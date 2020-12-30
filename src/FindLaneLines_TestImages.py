# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import HelperFunctions as hf

test_images = os.listdir("test_images/")
for i in range(len(test_images)):
  # reading in an image
  image = mpimg.imread("test_images/" + test_images[i])

  # printing out some stats and plotting
  # print('This image is:', type(image), 'with dimensions:', image.shape)
  # plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


  # convert to grayscale image
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # apply gaussian blur
  kernel_size = 5
  gray_blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

  # find edges with canny edge detection
  canny_threshold_low = 50
  canny_threshold_high = 150

  output_canny = cv2.Canny(gray_blur, canny_threshold_low, canny_threshold_high)

  # mask with region of interest
  vertices = np.array([[[80,540], [920,540], [515,320], [445,320]]], dtype=np.int32) #both
  # vertices = np.array([[[80,540], [500,540], [500,320], [445,320]]], dtype=np.int32) #only left
  # vertices = np.array([[[500,540], [920,540], [515,320], [500,320]]], dtype=np.int32) #only right


  output_canny_roi = hf.region_of_interest(output_canny, vertices)

  # find long/same edge with hough transformation
  rho = 1
  theta = np.pi/180
  threshold = 1
  min_line_len = 8
  max_line_gap = 5

  [hough_roi, hough_lines] = hf.hough_lines(output_canny_roi, rho, theta, threshold, min_line_len, max_line_gap)

  # find left and right lane
  lanes = hf.extrapolate_lanes(hough_lines)
  image_lanes = np.copy(image)
  hf.draw_lines(image_lanes,lanes, thickness=10)
  # plt.imshow(image_lanes)

  # draw hough lines back to original image
  image_hough = hf.weighted_img(hough_roi, image, α=0.8, β=1., γ=0.)

  # plot all steps for one test image
  # plt.subplot(2,2,1)
  # plt.imshow(image)
  # plt.subplot(2,2,2)
  # plt.imshow(output_canny, cmap='gray')
  # plt.subplot(2,2,3)
  # plt.imshow(image_hough, cmap='gray')
  # plt.subplot(2,2,4)
  # plt.imshow(image_lanes)

  
  # plot output on all test images (canny + hough_roi)
  plt.subplot(3,4,2*i+1)
  plt.imshow(image, cmap='gray')
  plt.subplot(3,4,2*i+2)
  plt.imshow(image_hough)
  

plt.show()


