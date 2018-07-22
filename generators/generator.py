from sklearn.utils import shuffle
from data_access.image_repository import get_image_by_id
import numpy as np


def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []
            throttles_positions = []
            for image_meta in batch_samples:
                image = get_image_by_id(image_meta['imageId'])
                images.append(image)
                steering_angles.append(image_meta['steeringAngle'])
                throttles_positions.append(image_meta['throttle'])

            X = [images]
            y = [np.array(steering_angles), np.array(throttles_positions)]

            yield X, y