import os
import cv2
import FindLaneLine
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def main():
  fll = FindLaneLine.Image()

  test_videos = os.listdir("test_videos/") # 0 => yellow, 1 => challenge, 2 => white
  test_videos_input  = list()
  test_videos_output = list()

  for file_name in test_videos:
    test_videos_input.append(f"test_videos/{file_name}")
    test_videos_output.append(f"test_videos_output/{file_name}")

  # j = 0
  # clip = VideoFileClip(test_videos_input[j]).subclip(0,2)
  # clip_with_lanes = clip.fl_image(fll.process_image)
  # clip_with_lanes.write_videofile(test_videos_output[j], audio=False)

  for j, item in enumerate(test_videos):
    clip = VideoFileClip(test_videos_input[j])
    clip_with_lanes = clip.fl_image(fll.process_image)
    clip_with_lanes.write_videofile(test_videos_output[j], audio=False)

''' ### main() for test_images
def main():
  # print("Self Driving Car Engineer: Project 1) Finding Lane Lines")

  test_images = os.listdir("test_images/")

  ### pythonic loop => https://realpython.com/courses/how-to-write-pythonic-loops/
  #  for i,item  in enumerate(test_images):
    # test_images[i] = "test_images/" + test_images[i]

  ### list comprehension => https://realpython.com/list-comprehension-python/#how-to-supercharge-your-comprehensions
  test_images = ["test_images/" + item for item in test_images if ".jpg" in item] # list comprehension, with conditionalj

  # test_images = test_images[0:1]

  tst_imgs = list()

  for i, path in enumerate(test_images):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    tst_imgs.append(FindLaneLine.Image())
    tst_imgs[i].process_image(img, 4)
'''

if __name__ == "__main__":
    main()

