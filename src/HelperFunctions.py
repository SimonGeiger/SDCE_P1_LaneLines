import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean,median


def grayscale(img):
    """ Applies the Grayscale transform
    This will return an image with only one color channel but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray') you should call plt.imshow(gray, cmap='gray') """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return [line_img, lines]


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def calculate_lane_cartesian(m, b, y):
    """
    This function outputs the x coordinate on a straight line (defined by theta and rho)
    """
    return int((y-b) / m)


def extrapolate_lanes(hough_lines, img):
    """
    This function takes the hough_lines and translates it into left and right lane by 
    1) calculating the slope m and y-intercept b for each hough line. 
    2) sorting into left lane and right lane by distiguishing positive and negative slopes
    3) determining median slope/y inercept for each left and right lane
    4) drawing red lines into empty image
    5) returning that image
    input: hough lines (output of cv2.HoughLines)
    output: image with detected lanes only (mask)
    """
    y_start = 540
    y_end = 340

    left_lane = [-1,-1,-1,-1]
    right_lane = [-1,-1,-1,-1]

    m_left = list()
    b_left = list()
    m_right = list()
    b_right = list()

    m_left_mean = 0.0
    m_right_mean = 0.0
    b_left_mean = 0.0
    b_right_mean = 0.0

    # analyze hough_lines to dermine left vs. right line
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        if  x2 != x1:
            # if y2 >= y1:
            m = (y2-y1) / (x2-x1)
            # else:
            #     m = (y1-y2) / (x1-x2)
            b = y1 - m * x1   
        else:
            m = 'inf' 
            b = 'n/a'

        if m == 'inf' or m == 0:
            # vertical or horizontal lane => can't be used
            continue


        m_wanted_left = 0.55
        m_wanted_right = -0.68
        m_tolerance = 0.25
        if m > (m_wanted_left - m_tolerance) and  m < (m_wanted_left + m_tolerance):
            # left line => / 
            m_left.append(m)
            b_left.append(b)
        elif m > (m_wanted_right - m_tolerance) and m < (m_wanted_right + m_tolerance):
            # right line => \ 
            m_right.append(m)
            b_right.append(b)

        else:
            # -0.5 < m < 0.5 => line to the left or to the right => not useful
            continue

    if len(m_left)>0:
        m_left_mean = mean(m_left) 
        b_left_mean = mean(b_left)
        left_lane = [calculate_lane_cartesian(m_left_mean,b_left_mean,y_start),y_start, # x_start, y_start, 
                     calculate_lane_cartesian(m_left_mean,b_left_mean,y_end),y_end] # x_end, y_end
    if len(m_right)>0:
        m_right_mean = mean(m_right)
        b_right_mean = mean(b_right)
        right_lane = [calculate_lane_cartesian(m_right_mean,b_right_mean,y_start),y_start, # x_start, y_start, 
                      calculate_lane_cartesian(m_right_mean,b_right_mean,y_end),y_end] # x_end, y_end

    # plot to display mean/median value within extracted lines
    # plt.subplot(2,1,1)
    # plt.hist(m_left)
    # plt.hist(m_right)
    # plt.axvline(m_left_mean, color='k', linestyle='dashed', linewidth=1)
    # plt.axvline(m_right_mean, color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(m_left_mean + 0.02, max_ylim*0.9, f"mean: {m_left_mean:.2f}")
    # plt.text(m_left_mean + 0.02, max_ylim*0.8, f"num: {len(m_left)}")
    # plt.text(m_right_mean + 0.02, max_ylim*0.9, f"mean: {m_right_mean:.2f}")
    # plt.text(m_right_mean + 0.02, max_ylim*0.8, f"num: {len(m_right)}")
    # plt.subplot(2,1,2)
    # plt.hist(b_left)
    # plt.hist(b_right)
    # plt.axvline(b_left_mean, color='k', linestyle='dashed', linewidth=1)
    # plt.axvline(b_right_mean, color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(b_left_mean + 2, max_ylim*0.9, f"mean: {b_left_mean:.2f}")
    # plt.text(b_left_mean + 2, max_ylim*0.8, f"num: {len(b_left)}")
    # plt.text(b_right_mean + 2, max_ylim*0.9, f"mean: {b_right_mean:.2f}")
    # plt.text(b_right_mean + 2, max_ylim*0.8, f"num: {len(b_right)}")
    # plt.show()

    lanes = [[left_lane], [right_lane]]

    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(lanes_img,lanes, thickness=10)

    return lanes_img