from tensorflow.python.lib.io import file_io
import os
base_path = 'gs://sacred-reality-201417-mlengine/models'


def save_model(file_name):
    with file_io.FileIO(file_name, mode='rb') as f:
        with file_io.FileIO(os.path.join(base_path, file_name), mode='w+') as output_f:
            output_f.write(f.read())


def get_filename(base_name):
    counter = 1
    while file_io.file_exists(generate_name(base_name, counter)):
        counter += 1
    return generate_name(base_name, counter)


def generate_name(name, counter):
    return '{}-{}'.format(name, counter)
