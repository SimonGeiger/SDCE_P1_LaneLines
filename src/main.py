import os
import FindLaneLine
import matplotlib.pyplot as plt

def main():
  # print("Self Driving Car Engineer: Project 1) Finding Lane Lines")

  test_images = os.listdir("test_images/")

  ### pythonic loop => https://realpython.com/courses/how-to-write-pythonic-loops/
  #  for i,item  in enumerate(test_images):
    # test_images[i] = "../test_images/" + test_images[i]

  ### list comprehension => https://realpython.com/list-comprehension-python/#how-to-supercharge-your-comprehensions
  test_images = ["test_images/" + item for item in test_images if ".jpg" in item] # list comprehension, with conditionalj
  
  # test_images = test_images[0:1]

  fll = list()

  # TODO: Read from Video
  for i, path in enumerate(test_images):
    fll.append(FindLaneLine.Image(path))
    fll[i].create_color_mask()
    fll[i].find_hough_lines_in_roi()
    fll[i].get_lanes()
    fll[i].print(3)


if __name__ == "__main__":
    main()

