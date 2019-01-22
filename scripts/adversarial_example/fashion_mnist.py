from keras.utils import to_categorical
from keras import callbacks
from keras.datasets import fashion_mnist
from scripts.deep import network_models
from src import saliency_function, layers
from keras.models import save_model, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

batch_size = 128
lasso = 0.
regularization = 5e-4
lr = .01
input_noise = .5

directory = './scripts/adversarial_example/info_ruido/'
network_name = 'wrn164'
model_filename = 'fashion_mnist_wrn164.h5'


def scheduler(epoch):
    if epoch < 10:
        return 0.1
    elif epoch < 20:
        return 0.02
    elif epoch < 30:
        return 0.004
    else:
        return 0.0008


def learning_rate(accuracy):
    if accuracy < 1e-7:
        return 100.
    elif accuracy < 1e-5:
        return 10.
    elif accuracy < 1e-3:
        return 1.
    elif accuracy < .05:
        return .1
    elif accuracy > .4:
        return .01
    else:
        return .001


def load_dataset(normalize=False):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(float)
    x_test = np.expand_dims(x_test, axis=-1).astype(float)
    generator = ImageDataGenerator(
        width_shift_range=4./28,
        height_shift_range=4./28,
        fill_mode='nearest',
        horizontal_flip=True
    )
    y_train = np.reshape(y_train, [-1, 1])
    y_test = np.reshape(y_test, [-1, 1])
    if normalize:
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-6] = 1.
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    else:
        x_train /= 255.
        x_test /= 255.
        mean = 0.
        std = 1.
    output = {
        'train': {
            'data': x_train,
            'label': y_train
        },
        'test': {
            'data': x_test,
            'label': y_test
        },
        'generator': generator,
        'mean': mean,
        'std': std
    }
    return output


def sample_images(samples, filename, show_diff=False):
    samples = np.asarray(samples)
    r, c = samples.shape[:2]
    fig, axs = plt.subplots(r, c, squeeze=False)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if j == 0 or not show_diff:
                image = samples[i, j][..., 0]
                image[image > .2] = 1.
                # normalize
                # image = (image - image.min()) / (image.max() - image.min())
                axs[i, j].imshow(image, cmap='gray')
            else:
                image_diff = np.abs(samples[i, j][..., 0] - samples[i, 0][..., 0])
                # image_diff = .5 * image_diff + .5
                axs[i, j].imshow(image_diff, cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(filename)
    plt.close()


def main():

    dataset = load_dataset()

    train_data = np.asarray(dataset['train']['data'])
    train_labels = dataset['train']['label']
    num_classes = len(np.unique(train_labels))

    test_data = np.asarray(dataset['test']['data'])
    test_labels = dataset['test']['label']

    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    if os.path.exists(directory + model_filename):
        model = load_model(
            directory + model_filename,
            custom_objects={'GaussianNoise': layers.GaussianNoise, 'Mask': layers.Mask}
        )
    else:
        model_kwargs = {
            'nclasses': num_classes,
            'lasso': lasso,
            'regularization': regularization,
            'input_noise': input_noise
        }

        generator = dataset['generator']
        generator_kwargs = {
            'batch_size': batch_size
        }

        model = network_models.wrn164(train_data.shape[1:], **model_kwargs)
        model.fit_generator(
            generator.flow(train_data, train_labels, **generator_kwargs),
            steps_per_epoch=train_data.shape[0] // batch_size, epochs=40,
            callbacks=[
                callbacks.LearningRateScheduler(scheduler)
            ],
            validation_data=(test_data, test_labels),
            validation_steps=test_data.shape[0] // batch_size,
            verbose=2
        )
        if not os.path.isdir(directory):
            os.makedirs(directory)
        save_model(model, directory + model_filename)

    model.saliency = saliency_function.get_saliency('categorical_crossentropy', model, reduce_func=None, use_abs=False)

    samples = []
    for i in range(1, 2):
        print('label', i)
        pos = np.where(test_labels[:, i] > 0.)[0]
        np.random.seed(42)
        p = pos[np.random.randint(pos.shape[0])]
        # predictions = model.predict(test_data[pos])
        # label_pred = np.argmax(predictions, axis=-1)
        # p = np.where(label_pred == i)[0]
        # p_min = np.argmin(predictions[p, i])
        # p = pos[p[p_min]]
        sample = test_data[p]
        i_samples = [sample]
        for label in range(num_classes):
            print('label', i, label)
            label_sample = sample.copy().astype(float)
            cat_label = to_categorical([label], num_classes=num_classes)
            prediction = model.predict(np.asarray([label_sample]))[0]
            while prediction[label] < .95:
                saliency = model.saliency([[label_sample], cat_label, 0])[0][0]
                saliency /= np.max(np.abs(saliency))
                label_sample += lr * saliency
                label_sample[label_sample < 0.] = 0.
                label_sample[label_sample > 1.] = 1.
                prediction = model.predict(np.asarray([label_sample]))[0]
                print(label, prediction[label])
            i_samples.append(label_sample)
            # for c, s in enumerate(i_samples):
            #     sample_images([[s]], filename=directory + 'iimage_' + str(i) + '_' + str(c) + '.png')
            # sample_images([i_samples], filename=directory + 'iimage_' + str(i) + '.png')
        samples.append(i_samples)

    sample_images(samples, filename=directory + 'image.png')
    sample_images(samples, filename=directory + 'image_diff.png', show_diff=True)


if __name__ == '__main__':
    os.chdir('../../')
    main()



