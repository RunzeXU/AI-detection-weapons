# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:58:50 2020

@author: 28771
"""

import time
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
start = time.time()
 
def smooth(x, window_len=13, window='hanning'):
    print(len(x), window_len)
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x
 
    
if __name__ == "__main__":
    print(sys.executable)
    #Setting fixed threshold criteria
    USE_THRESH = False
    #fixed threshold value
    THRESH = 0.6
    #Setting fixed threshold criteria
    USE_TOP_ORDER = False
    #Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    #Number of top sorted frames
    NUM_TOP_FRAMES = 50
     
    #Video path of the source file
    videopath = 'C:\\Users\\28771\\models\\research\\object_detection\\test_images1\\video7.mp4'
    #Directory to store the processed frames
    dir = './extract_testknife/'
    #smoothing window size
    len_window = int(30)
    
    
    print("target video :" + videopath)
    print("frame save directory: " + dir)
    # load video and compute diff between frames
    cap = cv2.VideoCapture(str(videopath)) 
    curr_frame = None
    prev_frame = None 
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0 
    while(success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
        success, frame = cap.read()   
    cap.release()
    
    # compute keyframe
    keyframe_id = set()
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id.add(keyframe.id) 
            
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                keyframe_id.add(frames[i].id)   
                
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id.add(frames[i - 1].id)
            
        plt.figure(figsize=(40, 20))
        plt.locator_params(numticks=100)
        plt.title('Mean inter-frame difference intensity',fontsize=60)
        plt.grid()
        plt.plot(sm_diff_array)
        plt.xlabel('Frame',fontsize=60)
        plt.ylabel('internsity',fontsize=60)
        plt.yticks(fontproperties = 'Times New Roman', size = 50)
        plt.xticks(fontproperties = 'Times New Roman', size = 50)
        plt.savefig(dir + 'plot.png')

    
    # save all keyframes as image
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    keyframes = []
    success, frame = cap.read()
    index = 0
    while(success):
        
        if index in keyframe_id:
            picturename = "keyframe_" + str(index) + ".jpg"
            cv2.imwrite(dir + picturename, frame)
            keyframe_id.remove(index)
        index = index + 1
        success, frame = cap.read()
        
    cap.release()
end =  time.time()
print("Execution Time: ", end - start)