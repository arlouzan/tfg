
import os
import sys
from collections import deque
import agent_model
import cv2
import numpy as np
import torch


import Video_Loader
#from models.model_utils import create_model, load_pretrained_model
#from utils.post_processing import post_processing
#from utils.misc import time_synchronized
VIDEO_PATH = "/dataset/test/videos"
INPUT_SIZE = (200, 400)
NUM_FRAMES_SEQUENCE = 9
RESULT_DIR = "/result_dir"
TRAINED_MODEL = "/models"


def demo(configs):
    self.configs = configs
    video_loader = Video_Loader(VIDEO_PATH, INPUT_SIZE, NUM_FRAMES_SEQUENCE)
    result_filename = os.path.join(RESULT_DIR, 'results.txt')
    num_frames_sequence = NUM_FRAMES_SEQUENCE
    frame_rate = video_loader.video_fps
    frame_dir = os.path.join(RESULT_DIR, 'frame')
    model_dir = TRAINED_MODEL
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)
            
    # model
    env = Enviroment()
    input_size = env.get_observation_space_values()
    model = DQNAgent()
    print("Getting the model....")
    assert model_dir is not None, "Need to load the pre-trained model"
    model = keras.models.load_model(model_dir)
    print("Model loaded")

    model.evaluate()
    middle_idx = int(num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1920, 1080
    #w_resize, h_resize = 320, 128
    #w_ratio = w_original / w_resize
    #h_ratio = h_original / h_resize
    with torch.no_grad():
        for count, imgs in video_loader:
            print(count)
            img = imgs
            # We convert it to grayscale to pass it throught the network
            img_gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            # Expand the first dim
            resized_imgs = img_gray.reshape(model.get_reshape())
            action = np.argmax(agent.get_qs(resized_imgs))
            env.step(action)
            x1, x2, y1, y2 = env.get_camera_coordinates()
            # We calculate the center 	
            center_x = x2 - x1 / 2
            center_y = y2 - y1 / 2

            # Get infor of the (middle_idx + 1)th frame
            if len(queue_frames) == middle_idx + 1:
                ploted_img = plot_detection(img, x1, x2, y1, y2)
                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
                if configs.show_image:
                    cv2.imshow('ploted_img', ploted_img)
                    cv2.waitKey(10)
                if configs.save_demo_output:
                    cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)
            queue_frames.append(frame_pred_infor)

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx, t2 - t1))

    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(
            os.path.join(configs.frame_dir), output_video_path)
        os.system(cmd_str)


def plot_detection(img, ball_pos, seg_img, events):
    """Show the predicted information in the image"""
    zoom = np.array([])
    zoom = np.array(img[y1:y2, x1:x2])
    return zoom

class configs(self):
    self.show_image = False
    self.save_demo_dir = RESULT_DIR
    self.frame_dir = RESULT_DIR + "/frame"
    self.save_demo_output= True
    
    
if __name__ == '__main__':
    configs = configs()
    demo(configs=configs)
