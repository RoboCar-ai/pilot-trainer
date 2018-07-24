from sklearn.utils import shuffle
from data_access.image_repository import get_image_by_id
import numpy as np
import time


def generator(samples, batch_size):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            t1 = time.time()
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []
            throttles_positions = []

            for image_meta in batch_samples:
                image = get_image_by_id(image_meta['imageId'])
                images.append(image)
                steering_angles.append(image_meta['steeringAngle'])
                throttles_positions.append(image_meta['throttle'])

            X = np.array(images)
            y = [np.array(steering_angles), np.array(throttles_positions)]
            t2 = time.time()
            print('elapsed time to generate data for batch:', t2 - t1)
            yield X, y
