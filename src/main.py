import os
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML

TEST_IMAGE_INPUT = 1
VIDEO_INPUT      = 2

def main(mode = VIDEO_INPUT):

  if mode == TEST_IMAGE_INPUT:
    ### for test_images

    # print("Self Driving Car Engineer: Project 1) Finding Lane Lines")


    ### pythonic loop => https://realpython.com/courses/how-to-write-pythonic-loops/
    #  for i,item  in enumerate(test_images):
      # test_images[i] = "test_images/" + test_images[i]

    ### list comprehension => https://realpython.com/list-comprehension-python/#how-to-supercharge-your-comprehensions
    # test_images = ["test_images/" + item for item in test_images if ".jpg" in item] # list comprehension, with conditionalj

    test_images = os.listdir("test_images/")
    test_images_input  = list()
    test_images_output = list()

    for file_name in test_images:
      test_images_input.append(f"test_images/{file_name}")
      test_images_output.append(f"test_images_output/{file_name}")

    tst_imgs = list()
    print(dir())

    for i, path in enumerate(test_images_input):
      img = cv2.imread(path,cv2.IMREAD_COLOR)
      tst_imgs.append(Image())
      output_img = tst_imgs[i].process_image(img, 4)
      cv2.imwrite(test_images_output[i], output_img)

  elif mode == VIDEO_INPUT:
    fll = Image()

    test_videos = os.listdir("test_videos/") # id = 0 => yellow, id = 1 => challenge, id = 2 => white
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

if __name__ == "__main__":
  from Image import Image 
  # main(TEST_IMAGE_INPUT)
  main(VIDEO_INPUT)
else:
  from src.Image import Image