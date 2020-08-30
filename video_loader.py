import os
from collections import deque

import cv2
import numpy as np


class Video_Loader:
    """The loader for demo with a video input"""

    def __init__(self, video_path = "/dataset/test/videos", input_size=(320, 128), num_frames_sequence=9):
        assert os.path.isfile(video_path), "No video at {}".format(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = input_size[0]
        self.height = input_size[1]
        self.count = 0
        self.num_frames_sequence = num_frames_sequence
        print('Length of the video: {:d} frames'.format(self.video_num_frames))

        self.images_sequence = deque(maxlen=num_frames_sequence)
        self.get_first_images_sequence()

    def get_first_images_sequence(self):
        # Load (self.num_frames_sequence - 1) images
        while (self.count < self.num_frames_sequence):
            self.count += 1
            ret, frame = self.cap.read()  # BGR
            assert ret, 'Failed to load frame {:d}'.format(self.count)
            self.images_sequence.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height)))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image

        ret, frame = self.cap.read()  # BGR
        assert ret, 'Failed to load frame {:d}'.format(self.count)
        self.images_sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgs = np.dstack(self.images_sequence)  # (200, 400, 1)
        # Transpose (H, W, C) to (C, H, W) --> fit input of DQNAgent model
        resized_imgs = resized_imgs.transpose(2, 0, 1)  # (1, 200, 400)

        return self.count, resized_imgs

    def __len__(self):
        return self.video_num_frames - self.num_frames_sequence + 1 # number of sequences
