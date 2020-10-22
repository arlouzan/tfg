import os
import agent_model_windows
import glob
import cv2
import numpy as np
from agent_model_windows import Enviroment,DQNAgent
import keras
from collections import deque

#from models.model_utils import create_model, load_pretrained_model
#from utils.post_processing import post_processing
#from utils.misc import time_synchronized


base_dir =os.getcwd()
target_dir = os.path.join(base_dir,"dataset","test","images","test_1")
annotation_dir=os.path.join(base_dir,"dataset","test","annotations","test_1")
video_path = os.path.join(base_dir,"dataset","test","videos","test_1.mp4")
INPUT_SIZE = (200, 400)
NUM_FRAMES_SEQUENCE = 9
RESULT_DIR = os.path.join(base_dir,"result_dir")
video_result_dir = os.path.join(RESULT_DIR,"video")
result_frame_dir = os.path.join(RESULT_DIR,"frame")
TRAINED_MODEL = os.path.join(base_dir,"models")
model_dir = os.path.join(TRAINED_MODEL,"DQN_Agent__-4900788.00max_-5513468.10avg_-5971082.00min__1603338001.model")
shape=(1,1080,1920,3)

def max_pos(lst):
    sublst = lst[0]
    max_index = np.argmax(sublst)
    return max_index
    

class Video_Loader:
    """The loader for demo with a video input"""

    def __init__(self, video_path = video_path, input_size=(200, 400), num_frames_sequence=9):
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
            frame_aux = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_img= frame_aux.reshape(shape)
            self.images_sequence.append(resized_img)


def demo(configs):
    
	# model
    env = Enviroment()
    input_size = env.get_observation_space_values()
    agent = DQNAgent()
    reshape = agent.get_reshape()
    print(reshape)
    print("Getting the model....")
    assert model_dir is not None, "Need to load the pre-trained model"
    model = keras.models.load_model(model_dir)
    print("Model loaded")

    configs = configs
    video_loader = Video_Loader(video_path, input_size, NUM_FRAMES_SEQUENCE)
    result_filename = os.path.join(RESULT_DIR, 'results.txt')
    num_frames_sequence = NUM_FRAMES_SEQUENCE
    frame_rate = video_loader.video_fps
    frame_dir = os.path.join(RESULT_DIR, 'frame')
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)
            
    middle_idx = int(num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1920, 1080
    #w_resize, h_resize = 320, 128
    #w_ratio = w_original / w_resize
    #h_ratio = h_original / h_resize
    while(video_loader.cap.isOpened()):
        count, frame = video_loader.cap.read()
        if count:
            img = frame
            x1, x2, y1, y2 = env.get_camera_coordinates()
            # We convert it to grayscale to pass it throught the network
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_imgs = np.array(img_gray[y1:y2, x1:x2])
            # Expand the first dim
            resized_imgs = resized_imgs.reshape(reshape)
            #If I load the model 1the conv_2d need a transpose matrix
            transpose_imgs = resized_imgs.transpose(0,1, 2, 3)
            action = np.argmax(agent.get_qs(transpose_imgs))
            action = agent.get_qs(resized_imgs)
            print('Action {}-'.format(action))
            test = max_pos(action)
            print('Test_action {}-'.format(test))
            env.steps_demo(test)
            print('Coordinates of the camera {}-'.format(env.get_camera_coordinates()))
            # We calculate the center 	
            center_x = x2 - x1 / 2
            center_y = y2 - y1 / 2


            ploted_img = plot_detection(frame, x1, x2, y1, y2)
            ploted_img_bgr = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
            if configs.show_image:
                cv2.imshow('ploted_img', ploted_img_bgr)
                cv2.waitKey(10)
            cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img_bgr)
            queue_frames.append(ploted_img)
            frame_idx += 1
            print('Done frame_idx {}-'.format(frame_idx))
        
        else:
            break

    if configs.output_format == 'video':
        print("Doing the video")
        output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(
            os.path.join(configs.frame_dir), output_video_path)
        os.system(cmd_str)
        print("Finish")


def making_video():
    print("Doing the video")
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter('video.mp4', fourcc, 1,(400,200))

    for j in range(0,9589):
        numbr = str(j)
        add_0 = 6-len(numbr)
        zero = ''
        for i in range(0,add_0):
            zero = zero + '0'
        full_strng = result_frame_dir +  "/"  + zero + numbr +'.jpg'
        print(full_strng)
        img = cv2.imread(full_strng)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    print("Finish")

def plot_detection(img, x1, x2, y1,y2):
    """Show the predicted information in the image"""
    zoom = np.array([])
    zoom = np.array(img[y1:y2, x1:x2])
    return zoom

class configs():
    def __init__(self):
        self.show_image = False
        self.save_demo_dir = RESULT_DIR
        self.frame_dir = RESULT_DIR + "/frame"
        self.save_demo_output= True
        self.output_format = 'video'

    
if __name__ == '__main__':
    configs = configs()
    #demo(configs=configs)
    making_video()