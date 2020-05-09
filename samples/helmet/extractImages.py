import sys
import argparse
import os
import glob 
from tqdm import tqdm

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut, every=10):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print ('Video has {} fps and {} total frames'.format(fps, frameCount))
    
    count = 0
    # success = True
    pbar = tqdm(total = frameCount)
    while success:
      if (count % every) == 0 :
          # print ('Read a new frame: ', count+1, success)
          outFile = os.path.join(pathOut, os.path.splitext(os.path.split(file)[1])[0] + "_frame%d.jpg" % count)
          writeStatus = cv2.imwrite( outFile, image)     # save frame as JPEG file
          if not writeStatus :
              print ('imwrite problem for writting ', outFile)
              break
      success,image = vidcap.read()
      count += 1
      pbar.update(1)
    pbar.close()

if __name__=="__main__":

    a = argparse.ArgumentParser(description='Video-2-Image converter : Extract every nth frame from a video input file. For eg., python extractImages3 --pathIn .\datasets\*.mp4 --pathOut .\test_frames\extracted_ --every 5')
    a.add_argument("--pathIn", help="path file pattern to video. For eg. .\datasets\*.mp4", required=True,) #default='.\\iphone X1maxPro\\20200315_033514000_iOS.mov')
    a.add_argument("--pathOut", help="path to store extracted images with file prefix. For eg., .\test_frames\extracted_", required=True,) #default='.\\test_frames\\20200315_033514000_iOS')
    a.add_argument("--every", help="every nth frame to be captured. ",
                   default=10)
    args = a.parse_args()
    print(args)
    print("Converting Video to Frames..")
    vidFiles = glob.glob(args.pathIn)
    if len(vidFiles) == 0 :
        print(' No video files found')
        exit(2)
    else :
        for file in vidFiles :
            print ('Processing Video file ', file)
            extractImages(file, args.pathOut, int(args.every))