import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from os.path import isfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.layers as kl


def getPaths(path, ASCII_list):
    dir_path = path + '/train_'
    images = []
    labels = []
    signs_iterated = 0
    limit_of_examples = 12000
    for dec_sign in ASCII_list:
        file_iter = 0
        file = (5 - len(str(file_iter))) * '0' + str(file_iter) + '.png'
        full_path = dir_path + str(format(dec_sign, 'x')) + '/train_' + str(format(dec_sign, 'x')) + '_' + file
        while isfile(full_path) and file_iter < limit_of_examples:
            images.append(full_path)
            labels.append(signs_iterated)
            file_iter += 1
            file = (5 - len(str(file_iter))) * '0' + str(file_iter) + '.png'
            full_path = dir_path + str(format(dec_sign, 'x')) + '/train_' + str(format(dec_sign, 'x')) + '_' + file
        print(signs_iterated)
        signs_iterated += 1

    return images, labels, limit_of_examples


def getTrainedModel(path, ASCII_list):
    ver = 61
    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []
    histories = []
    links, links_labels, n_examples = getPaths(path, ASCII_list)
    batch_size = 256
    epochs = 8
    model = buildModel(len(ASCII_list))

    trainSamples, valSamples, trainLabels, valLabels = train_test_split(links, links_labels, test_size=0.1)

    images = []
    for i in range(len(trainSamples)):
        img = load_img(trainSamples[i], color_mode='grayscale')
        img = img_to_array(img)
        img = cv2.resize(img, (64, 64))
        img = (img - 255) * (-1)
        images.append(img)
        print("{}/{}".format(i + 1, len(trainSamples)))
    images = np.array(images)
    trainSamples = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    trainLabels = to_categorical(trainLabels, num_classes=len(ASCII_list))

    images = []
    for i in range(len(valSamples)):
        img = load_img(valSamples[i], color_mode='grayscale')
        img = img_to_array(img)
        img = cv2.resize(img, (64, 64))
        img = (img - 255) * (-1)
        images.append(img)
        print("{}/{}".format(i + 1, len(valSamples)))
    images = np.array(images)
    valSamples = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    valLabels = to_categorical(valLabels, num_classes=len(ASCII_list))

    with open(str(ver) + "_fit_" + str(n_examples) + "_" + str(epochs) + "_" + str(batch_size) + ".txt", 'a') as file:
        sys.stdout = file
        history = model.fit(trainSamples, trainLabels, epochs=epochs, batch_size=batch_size,
                            validation_data=(valSamples, valLabels), shuffle=True)
        histories.append(history)
        accuracy += history.history['accuracy']
        val_accuracy += history.history['val_accuracy']
        loss += history.history['loss']
        val_loss += history.history['val_loss']
        sys.stdout = sys.__stdout__

    with open(str(ver) + "_accuracy_and_loss_" + str(n_examples) + "_" + str(epochs) + "_" + str(batch_size) + ".txt",
              'w') as file:
        for e in range(epochs):
            # file.write("Part {}, epoch {}: accuracy {}, loss {}".format(p+1, e+1, accuracy[p+e], loss[p+e]))
            file.write(str(loss[e]) + "," + str(accuracy[e]) + "," + str(val_loss[e]) + "," + str(val_accuracy[e]) + "\n")

    model.save(str(ver) + "NIST_HTR_model_" + str(n_examples) + "_" + str(epochs) + "_" + str(batch_size))

    with open(str(ver) + "_model_" + str(n_examples) + "_" + str(epochs) + "_" + str(batch_size) + ".txt", 'w') as file:
        sys.stdout = file
        model.summary()
        sys.stdout = sys.__stdout__

    summarize(histories, ver, n_examples, epochs, batch_size)


def summarize(histories, ver, n_examples, epochs, batch_size):
    for i in range(len(histories)):
        plt.title('Loss')
        plt.plot(range(1, epochs+1), histories[i].history['loss'], color='green', label='train')
        plt.plot(range(1, epochs+1), histories[i].history['val_loss'], color='red', label='test')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.xlabel('epochs')
        plt.ylabel('loss')
    plt.savefig(str(ver) + "_Loss_" + str(n_examples) + "_" + str(epochs) + "_" + str(batch_size))
    plt.show()

    for i in range(len(histories)):
        plt.title('Accuracy')
        plt.plot(range(1, epochs+1), histories[i].history['accuracy'], color='green', label='train')
        plt.plot(range(1, epochs+1), histories[i].history['val_accuracy'], color='red', label='test')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
    plt.savefig(str(ver) + "_Accuracy_" + str(n_examples) + "_" + str(epochs) + "_" + str(batch_size))
    plt.show()


def buildModel(num_classes):
    model = Sequential()
    model.add(kl.experimental.preprocessing.Rescaling(1 / 255, input_shape=(64, 64, 1)))
    model.add(kl.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(kl.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(kl.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(kl.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(kl.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(kl.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(kl.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(kl.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(kl.Dropout(0.2))
    model.add(kl.Flatten())
    #model.add(kl.Dense(256, activation="relu"))
    model.add(kl.Dense(128, activation="relu"))
    model.add(kl.Dense(num_classes, activation="softmax"))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def normalize(img):
    img = cv2.GaussianBlur(img, (1, 1), 0)
    target_size = (64, 64)
    img = img.astype(float)
    height, width = img.shape
    factor = min(target_size[0] / height, target_size[1] / width)
    new_height = (target_size[0] - height * factor) / 2
    new_width = (target_size[1] - width * factor) / 2

    M = np.float32([[factor, 0, new_width], [0, factor, new_height]])
    target_matrix = np.ones(shape=target_size) * 255
    img = cv2.warpAffine(img, M, dsize=target_size[::-1], dst=target_matrix, borderMode=cv2.BORDER_TRANSPARENT)
    img = (img - 255) * (-1)
    low = np.min(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = min(255, img[i][j] * 1.6)

    # plt.imshow(img, cmap='gray')
    # plt.show()
    img = img_to_array(img)
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))

    return img


def focus(img, doubt):
    diff = np.max(img) - np.min(img)
    boundary = np.max(img) - 0.2 * diff
    boundary = doubt
    encountered = False
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < boundary:
                encountered = True
                up = i - 1
                break
        if encountered:
            break

    encountered = False
    for i in range(img.shape[0] - 1, -1, -1):
        for j in range(img.shape[1]):
            if img[i][j] < boundary:
                encountered = True
                down = i + 1
                break
        if encountered:
            break

    letter = img[up:down]
    #print(letter.shape)
    margin = int(letter.shape[0] * 0.5)
    #print(margin)
    letter_ext = np.ones((letter.shape[0] + margin * 2, letter.shape[1] + margin * 2)) * 255
    letter_ext[margin:-margin, margin:-margin] = letter

    return letter_ext


def split(img, doubt):
    letter_list = []

    count = False
    for x in range(0, img.shape[1]):
        contains_letter = False

        if count:
            for y in range(0, img.shape[0]):
                if img[y][x] < doubt:
                    contains_letter = True

            if not contains_letter:
                letter = img[0:img.shape[0], column_num_1-1:x+1]
                letter = focus(letter, doubt)
                letter_list.append(letter)
                count = False

        if not count:
            for y in range(0, img.shape[0]):
                if img[y][x] < doubt:
                    contains_letter = True
                    count = True
                    column_num_1 = x

    return letter_list


if __name__ == '__main__':
    ASCII_list = list(range(ord('A'), ord('Z') + 1))
    #ASCII_list += list(range(ord('a'), ord('z') + 1))
    #print(ASCII_list)
    #getTrainedModel('./NIST', ASCII_list)

    img = cv2.imread('./tests/test3.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.show()
    word = split(img, 210)

    model = load_model('61NIST_HTR_model_12000_8_256')
    predicted_word = ''
    prediction_conf = 1
    for letter in word:
        # plt.imshow(letter, cmap='gray')
        # plt.show()
        letter = normalize(letter)
        results = model.predict(letter)
        result = np.argmax(results)
        predicted_word += chr(ASCII_list[result])
        prediction_conf *= max(results[0])
        print(chr(ASCII_list[result]), str(max(results[0])))
    print("The most probable word is: "+predicted_word)
