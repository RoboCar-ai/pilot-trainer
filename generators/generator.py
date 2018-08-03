from sklearn.utils import shuffle
import numpy as np


def generator(samples_x, samples_y,  batch_size):
    num_samples = len(samples_x)
    while True:
        shuffle(samples_x, samples_y)
        for offset in range(0, num_samples, batch_size):
            images = samples_x[offset:offset + batch_size]
            batch_samples_y = samples_y[offset:offset + batch_size]

            steering_angles = []
            throttles_positions = []

            for image_meta in batch_samples_y:
                steering_angles.append(image_meta['steeringAngle'])
                throttles_positions.append(image_meta['throttle'])

            X = np.array(images)
            y = [np.array(steering_angles), np.array(throttles_positions)]
            yield X, y
