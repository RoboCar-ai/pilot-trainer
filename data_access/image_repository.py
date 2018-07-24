import json
from tensorflow.python.lib.io import file_io
from PIL import Image
from io import BytesIO
import numpy as np

base_path = 'gs://sacred-reality-201417-mlengine/data'


def get_meta_by_capture(capture_name):
    """
    Loads data from cloud storage into a List.
    :return: List of Image MetaData
    """
    print('loading image metadata from GCP')

    image_meta_paths = file_io.get_matching_files('{}/{}/*.json'.format(base_path, capture_name))
    data = []
    num_samples = len(image_meta_paths)
    print('loading {} image metadata samples'.format(num_samples))
    next_completion_interval = .1
    for i, path in enumerate(image_meta_paths):
        with file_io.FileIO(path, mode='r') as f:
            data.append(json.loads(f.read()))
        completion_percentage = (i + 1) / num_samples
        if completion_percentage >= next_completion_interval:
            print("image metadata load is {0:.0%} complete".format(completion_percentage))
            next_completion_interval += .1

    print('num of records: {}'.format(len(data)))
    return data


def get_image_by_id(id):
    with file_io.FileIO('{}/{}.jpg'.format(base_path, id), mode='rb') as f:
        return np.array(Image.open(BytesIO(f.read())))

