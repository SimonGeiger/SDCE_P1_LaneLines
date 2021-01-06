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

  tst_imgs = list()

  # TODO: Read from Video
  for i, path in enumerate(test_images):
    tst_imgs.append(FindLaneLine.Image(path))
    tst_imgs[i].process_image()

if __name__ == "__main__":
    main()

