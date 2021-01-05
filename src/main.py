import os
import FindLaneLine

def main():
  print("Self Driving Car Engineer: Project 1) Finding Lane Lines")

  test_images = os.listdir("test_images/")

  ### pythonic loop => https://realpython.com/courses/how-to-write-pythonic-loops/
  #  for i,item  in enumerate(test_images):
    # test_images[i] = "../test_images/" + test_images[i]

  ### list comprehension => https://realpython.com/list-comprehension-python/#how-to-supercharge-your-comprehensions
  test_images = ["test_images/" + item for item in test_images if ".jpg" in item] # list comprehension, with conditionalj

  # TODO: Read from Video
  path_to_image = test_images[4]

  fll = FindLaneLine.MyImage(path_to_image)
  fll.create_color_mask()
  fll.find_edges()

  fll.print(0.5)


if __name__ == "__main__":
    main()
