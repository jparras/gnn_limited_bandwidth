import keras
from keras.datasets import mnist
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import numpy as np
import os
from tqdm import tqdm


def load_data_balanced(n_of_workers, num_classes, min_size=32):
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Balanced dataset: the imported dataset is already randomized

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # Imagenet expects a 3-D image input: repeat the RGB channels
    #x_train = np.repeat(x_train, 3, -1)
    #x_test = np.repeat(x_test, 3, -1)
    #input_shape = (img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # split the data among workers
    train_data_per_worker = np.floor(x_train.shape[0] / n_of_workers).astype(np.int)
    test_data_per_worker = np.floor(x_test.shape[0] / n_of_workers).astype(np.int)

    x_train = x_train[0: train_data_per_worker * n_of_workers]
    y_train = y_train[0: train_data_per_worker * n_of_workers]
    x_test = x_test[0: test_data_per_worker * n_of_workers]
    y_test = y_test[0: test_data_per_worker * n_of_workers]

    x_train = list(x_train.reshape(
        (n_of_workers, train_data_per_worker, x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    y_train = list(y_train.reshape((n_of_workers, train_data_per_worker, y_train.shape[1])))
    x_test_global = x_test
    y_test_global = y_test
    x_test = list(x_test.reshape(
        (n_of_workers, test_data_per_worker, x_test.shape[1], x_test.shape[2], x_test.shape[3])))
    y_test = list(y_test.reshape((n_of_workers, test_data_per_worker, y_test.shape[1])))

    return x_train, y_train, x_test, y_test, x_test_global, y_test_global, input_shape


def load_data_unbalanced(n_of_workers, num_classes):
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Order the training and validation data, to have unbalanced datasets
    train_sort = np.argsort(y_train)
    y_train = y_train[train_sort]
    x_train = x_train[train_sort]
    test_sort = np.argsort(y_test)
    y_test = y_test[test_sort]
    x_test = x_test[test_sort]

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(n_of_workers, ' workers')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train_o = []
    y_train_o = []
    x_test_o = []
    y_test_o = []
    x_test_global = x_test
    y_test_global = y_test
    # split the data in to have unbalanced datasets:
    d_p_a = 8  # Number of digits that each agent sees
    if n_of_workers > 1:
        # Find the labels
        labels_train = np.argmax(y_train, axis=1)
        labels_test = np.argmax(y_test, axis=1)
        # Obtain the maximum number of samples per agent in a conservative way (there will be samples not assigned!)
        min_train = int(np.floor(np.amin(np.unique(labels_train, return_counts=True)[1]) / n_of_workers))
        min_test = int(np.floor(np.amin(np.unique(labels_test, return_counts=True)[1]) / n_of_workers))
        idx_train = n_of_workers * np.ones_like(labels_train, dtype=int)
        idx_test = n_of_workers * np.ones_like(labels_train, dtype=int)
        # For each worker, assign samples
        for w in range(n_of_workers):
            digits = np.random.choice(np.arange(num_classes), size=d_p_a, replace=False).tolist()  # Digits seen by the agent
            for d in digits:
                # Distribute the train indexes
                valid_indexes = np.intersect1d(np.where(labels_train == d)[0], np.where(idx_train == n_of_workers)[0])
                idx_train[np.random.choice(valid_indexes, size=min_train, replace=False)] = w
                valid_indexes = np.intersect1d(np.where(labels_test == d)[0], np.where(idx_test == n_of_workers)[0])
                idx_test[np.random.choice(valid_indexes, size=min_test, replace=False)] = w
            # Distribute values
            x_train_o.append(x_train[np.where(idx_train == w)[0]].reshape(
                (min_train * d_p_a, x_train.shape[1], x_train.shape[2], x_train.shape[3])
            ))
            y_train_o.append(y_train[np.where(idx_train == w)[0]].reshape(
                (min_train * d_p_a, y_train.shape[1])
            ))
            x_test_o.append(x_test[np.where(idx_test == w)[0]].reshape(
                (min_test * d_p_a, x_test.shape[1], x_test.shape[2], x_test.shape[3])
            ))
            y_test_o.append(y_test[np.where(idx_test == w)[0]].reshape(
                (min_test * d_p_a, y_test.shape[1])
            ))
    else:  # 1 worker is always the baseline: it has all data!
        x_train_o = list(x_train.reshape(
            (n_of_workers, x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])))
        y_train_o = list(y_train.reshape((n_of_workers, y_train.shape[0], y_train.shape[1])))
        x_test_o = list(x_test.reshape(
            (n_of_workers, x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])))
        y_test_o = list(y_test.reshape((n_of_workers, y_test.shape[0], y_test.shape[1])))
    return x_train_o, y_train_o, x_test_o, y_test_o, x_test_global, y_test_global, input_shape


def generate_model(input_shape, num_classes, output_dim=64):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def generate_autoencoder(input_dim, latent_dim, int_dim_1=512, int_dim_2=256):

    input = Input(shape=(input_dim,))
    encoded = Dense(int_dim_1, activation='relu')(input)
    encoded = Dense(int_dim_2, activation='relu')(encoded)
    encoded_out = Dense(latent_dim, activation='relu')(encoded)

    enc_input = Input(shape=(latent_dim, ))
    decoded = Dense(int_dim_2, activation='relu')(enc_input)
    decoded = Dense(int_dim_1, activation='relu')(decoded)
    decoded_out = Dense(input_dim, activation='linear')(decoded)

    encoder = Model(input, encoded_out)
    decoder = Model(enc_input, decoded_out)
    autoencoder = Model(input, decoder(encoded_out))

    # Custom loss
    from keras import backend as K

    def jensen_distance(y_true, y_pred):  # For gaussian data
        std_true = K.std(y_true)
        mean_true = K.mean(y_true)
        std_pred = K.std(y_pred)
        mean_pred = K.mean(y_pred)
        kld_1 = K.log(std_true / std_pred) + \
                K.exp(K.log(K.square(std_pred) + K.square(mean_pred - mean_true)) -
                      K.log(K.constant(2) * K.square(std_true))) - K.constant(0.5)
        kld_2 = K.log(std_pred / std_true) + \
                K.exp(K.log(K.square(std_true) + K.square(mean_true - mean_pred)) -
                      K.log(K.constant(2) * K.square(std_pred))) - K.constant(0.5)
        return K.log(kld_1 + kld_2) # Log Jensen distance for training

    def mean_squared_exp_error(y_true, y_pred):
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.mean(K.square(K.exp(y_pred) - K.exp(y_true)), axis=-1)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


def load_dataset_ae(data_train, data_test, w_sent=50000):
    fname = 'data_ae_' + str(data_train) + '_' + str(data_test) + '_' + str(w_sent) + '.npy'
    if not os.path.isfile(fname):
        # Generate dataset
        x_train, y_train, x_test, y_test, x_test_global, y_test_global, input_shape = load_data_balanced(1, 10)
        data = np.zeros((data_train + data_test, w_sent))
        print('Generating training vectors...')
        for e in tqdm(range(data_train + data_test)):  # Generate weights!!
            if e % 50 == 0:  # Restart classifier every 50 iterations
                classifier = generate_model(input_shape, 10)
            classifier.fit(x_train[0], y_train[0], batch_size=6000, epochs=1, verbose=0)
            weight_vector = np.concatenate([a.flatten() for a in classifier.get_weights()])
            if w_sent > 0:
                threshold = np.sort(np.square(weight_vector))[-w_sent]
                weight_vector = weight_vector[np.square(weight_vector) >= threshold]
                weight_vector = weight_vector[0: w_sent]
            data[e] = weight_vector
        np.save(fname, data)
    data = np.load(fname)
    id_vector = np.arange(data_train + data_test)
    np.random.shuffle(id_vector)
    x_train = data[id_vector[0: data_train], :]
    x_test = data[id_vector[data_train:], :]
    return x_train, x_test


def obtain_autoencoder(latent_dim, batch_size=256, data_train=5000, data_test=500,
                       epochs=25, int_dim_1=512, int_dim_2=256, w_sent=100, w_l=0):
    # Note: batch size / data train optimized for RTX 2080 Ti + 64 GB of RAM
    # Load the weights form a saved model, in order to obtain the input dim and AE dimension
    import h5py
    from keras.engine.saving import load_attributes_from_hdf5_group
    with h5py.File('mnist_feat_extractor.h5', mode='r') as f:
        layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        weight_values = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
            weight_values.extend([np.asarray(g[weight_name]) for weight_name in weight_names])
    weights = np.concatenate([a.flatten() for a in weight_values])

    input_dim = w_sent - 2 * w_l if w_sent > 0 else weights.size

    autoencoder, encoder, decoder = generate_autoencoder(input_dim, latent_dim,
                                                         int_dim_1=int_dim_1, int_dim_2=int_dim_2)
    fname_enc = 'encoder_trained_' + str(input_dim) + '_' + str(latent_dim) + '.h5'
    fname_dec = 'decoder_trained_' + str(input_dim) + '_' + str(latent_dim) + '.h5'

    if (not os.path.isfile(fname_dec)) or (not os.path.isfile(fname_enc)):
        print('Training autoencoder for ', input_dim, ' input features and ', latent_dim, ' latent features')
        # Generate data for training the autoencoder
        # Weights in our features extractor follow the next Gaussian distribution approx:
        mean_example = np.mean(weights)
        std = np.std(weights)
        if w_sent > 0:  # Pruning!!
            x_train = np.zeros((data_train, input_dim))
            x_test = np.zeros((data_test, input_dim))
            print('Obtaining training dataset...')
            x_train, x_test = load_dataset_ae(data_train, data_test)
            '''
            for i in tqdm(range(data_train)):
                mean = mean_example + np.random.uniform(low=mean_example - 2 * std, high=mean_example + 2 * std)
                #th = np.clip(np.square(np.random.normal(loc=mean, scale=std)), 0, std ** 2)
                th = np.random.uniform(low=std ** 2, high=4 * std ** 2)
                aux_data = np.random.normal(loc=mean, scale=std, size=5 * input_dim)
                aux_data = aux_data[np.square(aux_data) >= th]
                while aux_data.size < input_dim:
                    ad = np.random.normal(loc=mean, scale=std, size=5 * input_dim)
                    ad = ad[np.square(ad) >= th]
                    aux_data = np.concatenate([aux_data, ad])
                x_train[i] = aux_data[0: input_dim]
            print('Obtaining validation dataset...')
            for i in tqdm(range(data_test)):
                mean = mean_example + np.random.uniform(low=mean_example - 2 * std, high=mean_example + 2 * std)
                #th = np.clip(np.square(np.random.normal(loc=mean, scale=std)), 0, std ** 2)
                th = np.random.uniform(low=std ** 2, high=4 * std ** 2)
                aux_data = np.random.normal(loc=mean, scale=std, size=5 * input_dim)
                aux_data = aux_data[np.square(aux_data) >= th]
                while aux_data.size < input_dim:
                    ad = np.random.normal(loc=mean, scale=std, size=5 * input_dim)
                    ad = ad[np.square(ad) >= th]
                    aux_data = np.concatenate([aux_data, ad])
                x_test[i] = aux_data[0: input_dim]
        else:
            x_train = np.random.normal(loc=mean_example, scale=std, size=(data_train, input_dim))
            x_test = np.random.normal(loc=mean_example, scale=std, size=(data_test, input_dim))
        '''
        # The autoencoder reconstruction is helped by sorting the values (i.e., correlation)
        x_train = np.sort(x_train, axis=1)
        x_test = np.sort(x_test, axis=1)
        if w_l > 0:
            x_train = x_train[:, w_l: -w_l]
            x_test = x_test[:, w_l: -w_l]
        # Train the autoencoder
        h = autoencoder.fit(x_train, x_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=False,  # Shuffle is not needed, as data has been randomly generated
                            verbose=1,
                            validation_data=(x_test, x_test))
        # Plot histogram
        inp_data = x_test.flatten()
        rec_data = autoencoder.predict(x_test).flatten()
        bins = np.linspace(np.amin(x_test), np.amax(x_test), 100)
        plt.hist(inp_data, bins, alpha=0.5, label='Input')
        plt.hist(rec_data, bins, alpha=0.5, label='Reconstructed')
        plt.legend(loc='upper right')
        plt.yscale('log', nonposy='clip')
        plt.title('Histograms for mean = ' + str(mean_example) + ' and std = ' + str(std))
        tikz_save('autoencoder_training_' + str(input_dim) + '_' + str(latent_dim) + '.tikz',
                  figureheight='\\figureheight', figurewidth='\\figurewidth')
        plt.savefig('autoencoder_training_' + str(input_dim) + '_' + str(latent_dim) + '.png', bbox_inches='tight')
        plt.close()
        # Plot training loss
        plt.plot(h.history['loss'], 'b')
        plt.plot(h.history['val_loss'], 'r')
        plt.title('AE loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        tikz_save('autoencoder_loss_training_' + str(input_dim) + '_' + str(latent_dim) + '.tikz',
                  figureheight='\\figureheight', figurewidth='\\figurewidth')
        plt.savefig('autoencoder_loss_training_' + str(input_dim) + '_' + str(latent_dim) + '.png', bbox_inches='tight')
        plt.close()
        # Plot example of AE
        plt.plot(x_test[0], 'b')
        plt.plot(autoencoder.predict(np.reshape(x_test[0], [1, input_dim])).flatten(), 'r--')
        plt.title('AE example')
        plt.ylabel('Value')
        plt.xlabel('Weight')
        plt.legend(['Input', 'Reconstruction'], loc='best')
        tikz_save('autoencoder_example_' + str(input_dim) + '_' + str(latent_dim) + '.tikz',
                  figureheight='\\figureheight', figurewidth='\\figurewidth')
        plt.savefig('autoencoder_example_' + str(input_dim) + '_' + str(latent_dim) + '.png', bbox_inches='tight')
        plt.close()

        encoder.save_weights(fname_enc)
        decoder.save_weights(fname_dec)

    encoder.load_weights(fname_enc)
    decoder.load_weights(fname_dec)

    return encoder, decoder
