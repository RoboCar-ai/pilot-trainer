import json
from glob import glob
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

base_path = 'gs://sacred-reality-201417-mlengine/data'


# TODO: may be removed if not needed.
def get_meta_by_capture(capture_name):
    """
    Loads data from cloud storage into a List.
    :return: List of Image MetaData
    """
    print('loading image metadata from GCP')

    image_meta_paths = glob('{}/*.json'.format(capture_name))
    y = []
    X = []
    num_samples = len(image_meta_paths)
    print('loading {} image samples'.format(num_samples))
    next_completion_interval = .1
    for i, path in enumerate(image_meta_paths):
        with open(path, mode='r') as f:
            meta_data = json.loads(f.read())
            y.append(meta_data)
        X.append(get_image_by_id(capture_name, meta_data['imageId']))
        completion_percentage = (i + 1) / num_samples
        if completion_percentage >= next_completion_interval:
            print("image metadata load is {0:.0%} complete".format(completion_percentage))
            next_completion_interval += .1

    print('num of records: {}'.format(len(X)))
    return X, y


def get_image_by_id(capture_name, id):
    return cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(capture_name, id)), cv2.COLOR_BGR2RGB)