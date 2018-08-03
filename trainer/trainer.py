# external libs
import time
import os
import keras
from keras.layers import Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
import argparse
import subprocess

# internal imports
from data_access.image_repository import get_meta_by_capture
from generators.generator import generator
import data_access.model_repository as model_repository

BATCH_SIZE = 128


def load_from_repository(capture_name):
    return get_meta_by_capture(capture_name)


def get_model():
    drop = 0.1
    # First layer, input layer, Shape comes from camera.py resolution, RGB
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(drop)(x)

    # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # 64 features, 3px3p kernal window, 1wx1h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to
    # be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(drop)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle Connect every input with every output and output 15 hidden units. Use Softmax
    # to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    angle_out = Dense(1, name='angle_out')(x)

    # continuous output of throttle
    throttle_out = Dense(1, name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    return model


def compile_model(model):
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss={'angle_out': 'mse',
                        'throttle_out': 'mse'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 1.0})


def main():
    print('running trainer with dataset:', dataset_name)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=.0005,
                                               patience=5,
                                               verbose=True,
                                               mode='auto')
    load_start = time.time()
    X, y = load_from_repository(dataset_name)
    load_end = time.time()

    print('load time in minutes:', (load_end - load_start) / 60)
    print('total records', len(X))

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2)

    num_to_train = len(train_x)
    num_to_validate = len(val_x)

    print('total train records:', num_to_validate)
    print('total validation records:', num_to_validate)

    train_gen = generator(train_x, train_y, BATCH_SIZE)
    val_gen = generator(val_x, val_y, BATCH_SIZE)

    model = get_model()
    compile_model(model)
    model.summary()

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=num_to_train // BATCH_SIZE,
        epochs=10,
        verbose=True,
        validation_data=val_gen,
        callbacks=[early_stop],
        validation_steps=num_to_validate // BATCH_SIZE,
        workers=1,
        use_multiprocessing=False)

    print('saving model')
    model_name = model_repository.get_filename(dataset_name)
    model.save(model_name)
    model_repository.save_model(model_name)
    print('model saved {}'.format(model_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name',
                        help='name of dataset',
                        type=str)

    parser.add_argument('--job-dir',
                        help='ml engine workspace',
                        default=None,
                        type=str)

    args = parser.parse_args()
    dataset_name = args.dataset_name

    file_copy_start = time.time()
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    subprocess.call(['gsutil', '-m', 'cp', '-r', 'gs://sacred-reality-201417-mlengine/data/{}/*'.format(dataset_name), dataset_name])
    file_copy_end = time.time()

    print('file copy ET in Minutes:', (file_copy_end - file_copy_start) / 60)

    start = time.time()
    main()
    end = time.time()

    print('Training ET in minutes:', (end - start) / 60)
