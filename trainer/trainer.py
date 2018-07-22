# external libs

import io
from tensorflow.python.lib.io import file_io
from PIL import Image
import json
import time
import numpy as np
import keras
import os

# internal imports
from data_access.image_repository import get_meta_by_capture


def load_from_repository():
    return get_meta_by_capture('')


def main():
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=.0005,
                                               patience=5,
                                               verbose=True,
                                               mode='auto')

    data = load_from_repository()

    train_gen = generator(gen_records, cfg.BATCH_SIZE, 'model.h5')
    val_gen = generator(gen_records, cfg.BATCH_SIZE, 'model.h5')



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    print('ET:', end - start)