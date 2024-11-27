#!/usr/bin/env python3

import numpy as np
import cv2
import torch
from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from packages.drqv2 import CombinedModel

obs_shape = (9, 84, 84)
action_shape = (2, )
feature_dim = 50
hidden_dim = 1024

encoder_weights_path = "encoder_weights.pth"
actor_weights_path = "actor_weights.pth"

class PyTorchModel:
    def __init__(self):
        print("initializing PyTorch model")
        self.model = CombinedModel(obs_shape, action_shape, feature_dim, hidden_dim)
        print("load encoder")
        self.model.encoder.load_state_dict(torch.load(encoder_weights_path, weights_only=True))
        print("load actor")
        self.model.actor.load_state_dict(torch.load(actor_weights_path, weights_only=True))
        self.img_queue = np.zeros((3, 3, 84, 84))
        print("done innitialization")

    def get_wheel_velocities_from_image(self, img):
        # Resize img to (3, 84, 84)
        resized_img = cv2.resize(img, (84, 84)).transpose((2,0,1))
        print(f"shape of resized image after supposedly removing first image: {resized_img.shape}")

        print(f"shape of image queu before supposedly removing first image: {self.img_queue.shape}")        
        self.img_queue = self.img_queue[1:, ...]
        print(f"shape of image queu after supposedly removing first image: {self.img_queue.shape}")
        self.img_queue = np.vstack([self.img_queue, np.expand_dims(resized_img,0)])
        print(f"size of img queu after append: {self.img_queue.shape}")
        input = self.img_queue.reshape((-1, 84, 84))
        print(f"size of input: {input.shape}")
        print("calling model")
        mu, _ = self.model(torch.tensor(input, dtype=torch.uint8), 1)
        print("got model output")
        pwm = DifferentialPWM(left=mu[0], right=mu[1])
        raise
        return pwm


