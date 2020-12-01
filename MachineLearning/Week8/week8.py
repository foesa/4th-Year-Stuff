import numpy as np
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys


def conv_modeller(i):
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # change the below number to 5k,10k,20k,40k for (iii)
    n = 5000
    x_train = x_train[1:n]
    y_train = y_train[1:n]
    # x_test=x_test[1:500]; y_test=y_test[1:500]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    flat_arr = y_train.flatten()
    # zero_count = 0
    # one_count = 0
    # for i in flat_arr:
    #     if i == 0:
    #         zero_count = zero_count +1
    #     else:
    #         one_count = one_count + 1
    #
    # zero_arr = np.random.randint(10, size=4999).reshape(-1,1)
    # zero_arr = keras.utils.to_categorical(zero_arr, num_classes)
    # flat_arr = x_train.flatten().reshape(4999,3072)
    use_saved_model = False
    if use_saved_model:
        model = keras.models.load_model("cifar.model")
    else:
        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
        # model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2), padding='same'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2), padding='same'))
        # model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(i)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()

        batch_size = 128
        epochs = 20
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("cifar.model")
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss');
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.show()

    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1, y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1, y_pred))

    # y_pred = np.argmax(zero_arr, axis=1)
    # print(classification_report(y_train1, y_pred))
    # print(confusion_matrix(y_train1, y_pred))


def convolver(input_array, kernel):
    k_width = len(kernel)
    k_height = len(kernel[0])
    if k_height <= len(input_array[0]) and k_width <= len(input_array):
        shift_len = len(input_array) - k_width + 1
        out_arr = []
        for i in range(shift_len):
            for s in range(shift_len):
                temp = 0
                for f in range(k_width):
                    for j in range(k_height):
                        kern = kernel[f][j]
                        inp = input_array[f + i][j + s]
                        temp += kern * inp
                out_arr.append(temp)
        offset = (len(input_array) - k_width) + 1
        output = [out_arr[i:i + offset] for i in range(0, len(out_arr), offset)]
        return output


# inp = [[1, 2, 3, 4, 5], [1, 3, 2, 3, 10], [3, 2, 1, 4, 5], [6, 1, 1, 2, 2], [3, 2, 1, 5, 4]]
# kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
# k2 = [[0, -1, 0], [-1, 8, -1], [0, -1, 0]]
# print(convolver(inp, k2))
# k1 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
#
# im = Image.open('rsz_empty-pentagon.jpg')
# rgb = np.array(im.convert('RGB'))
# r = rgb[:, :, 0]
# output = np.array(convolver(r, k2))
# Image.fromarray(np.uint(output)).show()
#
# print(np.array_equal(convolver(r,k1)))
# s_vals = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# for i in s_vals:
#     conv_modeller(i)

conv_modeller(i=0.001)