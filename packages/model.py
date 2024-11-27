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
        self.model.encoder.load_state_dict(torch.load(encoder_weights_path))
        self.model.actor.load_state_dict(torch.load(actor_weights_path))
        self.img_queue = np.zeroes((3, 3, 84, 84))


    def get_wheel_velocities_from_image(self, img):
        # Resize img to (3, 84, 84)
        resized_img = cv2.resize(img, (3, 84, 84)) 
        
        self.img_queue = self.img_queue[1:, ...]
        self.img_queue = np.append(self.img_queue, resized_img)
        input = self.img_queue.reshape((-1, 84, 84))
        
        mu, _ = self.model(input, 1)

        pwm = DifferentialPWM(left=mu[0], right=mu[1])
        return pwm