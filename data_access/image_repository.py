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
    basepath = 'gs://sacred-reality-201417-mlengine/data'
    image_meta_paths = file_io.get_matching_files('{}/*.json'.format(basepath))
    data = []
    for path in image_meta_paths[100:110]:
        print('pulling from path', path)
        with file_io.FileIO(path, mode='r') as f:
            data.append(json.loads(f.read()))

    print(data)
    print('num of records: {}'.format(len(data)))
    return data


def get_image_by_id(id):
    with file_io.FileIO('{}/{}'.format(base_path, id), mode='rb') as f:
        return np.array(Image.open(BytesIO(f.read())))

