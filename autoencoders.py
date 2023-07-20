import numpy as np
import os
import pickle
from copy import deepcopy
import argparse
from tqdm import tqdm
from common import load_data_balanced, load_data_unbalanced, generate_model, generate_autoencoder, obtain_autoencoder


def argsparser():
    parser = argparse.ArgumentParser("MNIST with autoencoders")
    parser.add_argument('--now', help='Number of workers', type=int, default=2)
    parser.add_argument('--latentdim', help='Latent_dim', type=int, default=100)
    parser.add_argument('--w_sent', help='Weights sent (for pruning)', type=int, default=50000)
    parser.add_argument('--intdim1', help='Intermediate dim 1', type=int, default=512)
    parser.add_argument('--intdim2', help='Intermediate dim 2', type=int, default=128)
    parser.add_argument('--epochs', help='Training epochs', type=int, default=100)
    parser.add_argument('--task', help='Task description', type=str, default='train')
    parser.add_argument('--mode', help='Mode of the dataset', type=str, default='balanced')
    return parser.parse_args()


def train_autoencoder(latent_dim, int_dim_1, int_dim_2, w_sent, w_l=0):
    # First, set memory limit dynamically
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    _, _ = obtain_autoencoder(latent_dim=latent_dim, int_dim_1=int_dim_1, int_dim_2=int_dim_2, w_sent=w_sent, w_l=w_l)


def train(now, latent_dim, int_dim_1, int_dim_2, w_sent, mode, epochs=100, num_classes=10, batch_size=2048, alpha=0.5,
          w_l=250):
    save_file = 'results_auto_' + str(now) + '_mode_' + str(mode) + '.pickle'
    if not os.path.isfile(save_file):
        # First, set memory limit dynamically
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        print('Loading dataset...')
        if mode == 'balanced':
            load_data = load_data_balanced
        elif mode == 'unbalanced':
            load_data = load_data_unbalanced
        else:
            raise RuntimeError('Mode not recognized')
        x_train, y_train, x_test, y_test, x_test_global, y_test_global, input_shape = load_data(now, num_classes)
        print('Loading classificators...')
        classifier = [generate_model(input_shape, num_classes) for _ in range(now)]  # One per worker
        weights = classifier[0].get_weights()
        for w in range(now):
            classifier[w].set_weights(weights)
        print('Loading autoencoders...')
        encoder, decoder = obtain_autoencoder(latent_dim=latent_dim, int_dim_1=int_dim_1, int_dim_2=int_dim_2,
                                              w_sent=w_sent, w_l=w_l)

        # Prepare for training
        loss = np.zeros((epochs, now))
        accuracy = np.zeros((epochs, now))
        loss_global = np.zeros((epochs, now))
        accuracy_global = np.zeros((epochs, now))
        for e in tqdm(range(epochs)):
            #print('Training for epoch ', e + 1, ' of ', epochs)
            # Train each model one step on its data
            for w in range(now):
                # print('Training ', w, ' of ', now)
                classifier[w].fit(x_train[w], y_train[w], batch_size=batch_size, epochs=1, verbose=0,
                                  validation_data=(x_test[w], y_test[w]))
            if now > 1:
                # Transmit weights to other agent
                w_vector = []
                for w in range(now):
                    w_vector.append(classifier[w].get_weights())
                for w in range(now):
                    receiver_w = (w + 1) % now  # Index of worker that will receive the values
                    # Select the weights to transmit
                    sender_weights = w_vector[w]
                    weight_vector = np.concatenate([a.flatten() for a in sender_weights])
                    index_vector = np.arange(weight_vector.size)
                    order_idx = np.argsort(weight_vector)  # Order the weights
                    weights_tx = np.copy(weight_vector[order_idx])
                    index_tx = np.copy(index_vector[order_idx])  # Note: order_idx and index_tx are the same vector!
                    if w_sent > 0:
                        threshold = np.sort(np.square(weight_vector))[-w_sent]
                        id_aux = np.square(weights_tx) >= threshold
                        while np.sum(id_aux) > w_sent:  # Assertion
                            id_aux[np.where(id_aux == True)[0][0]] = False
                        weights_tx = weights_tx[id_aux]
                        index_tx = index_tx[id_aux]
                    weights_low = weights_tx[0: w_l]  # Weights with lowest values
                    weights_high = weights_tx[-w_l:]  # Weights with highest values
                    weights_cod = weights_tx[w_l: -w_l]  # Weigths to code in the AE
                    weights_decoded = decoder.predict(encoder.predict(np.reshape(weights_cod, [1, weights_cod.size])
                                                                      )).flatten()
                    weights_decoded = np.concatenate([weights_low, weights_decoded, weights_high])
                    #weights_decoded = weights_tx
                    #weights_decoded = decoder.predict(encoder.predict(np.reshape(weights_tx, [1, weights_tx.size])
                    #                                                  )).flatten()
                    if w_sent > 0:  # and 2 * w_l + w_sent < weights_tx.size:
                        weights_decoded[np.square(weights_decoded) < threshold] = 0
                    if w_sent > 0:
                        weights_rx = np.zeros(weight_vector.size)  # The receiver knows the number of weights!
                        for i in range(w_sent):
                            weights_rx[index_tx[i]] = weights_decoded[i]
                    else:
                        weights_rx = weights_decoded[np.argsort(index_tx)]  # Reorder the received weights!
                    weights_rec = deepcopy(sender_weights)  # To recover the weights in a proper structure
                    i = 0
                    for ida in range(len(weights_rec)):  # For each layer
                        idx = weights_rec[ida].size
                        weights_rec[ida] = np.zeros(weights_rec[ida].shape)
                        weights_rec[ida] = np.reshape(weights_rx[i : i + idx], weights_rec[ida].shape)
                        i = i + idx

                    # Update weights
                    receiver_weights = w_vector[receiver_w]
                    updated_weights = [(1 - alpha) * receiver_weights[i] + alpha * weights_rec[i]
                                       for i in range(len(receiver_weights))]
                    classifier[receiver_w].set_weights(updated_weights)

            # Obtain the score
            for w in range(now):
                # print('Obtaining the score ', w, ' of ', now)
                score = classifier[w].evaluate(x_test[w], y_test[w], verbose=0)
                score_global = classifier[w].evaluate(x_test_global, y_test_global, verbose=0)
                loss[e, w] = score[0]  # Loss
                accuracy[e, w] = score[1]  # Accuracy
                loss_global[e, w] = score_global[0]
                accuracy_global[e, w] = score_global[1]
            print('Finished epoch e = ', e + 1)
            print('Test loss:', loss[e, :], '; mean = ', np.mean(loss[e, :]))
            print('Test accuracy:', accuracy[e, :], '; mean = ', np.mean(accuracy[e, :]))
            print('Test loss global:', loss_global[e, :], '; mean = ', np.mean(loss_global[e, :]))
            print('Test accuracy global:', accuracy_global[e, :], '; mean = ', np.mean(accuracy_global[e, :]))

        data = {'loss': loss,
                'accuracy': accuracy,
                'loss_global': loss_global,
                'accuracy_global': accuracy_global,
                'epochs': epochs,
                'now': now,
                'latent_dim': latent_dim,
                'data_dims': [x_train[0].shape, x_test[0].shape],
                'mode': mode}
        with open(save_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Finished training for ', now, ' workers , ', mode, ' mode')
    else:
        print('Results not computed for ', now, ' workers , ', mode, ' mode')


if __name__ == '__main__':

    #train(4, 500, 1024, 512, 50000, 'unbalanced', epochs=200)
    args = argsparser()
    w_l = 250
    if args.task == 'getae':
        train_autoencoder(args.latentdim, args.intdim1, args.intdim2, args.w_sent, w_l=w_l)
    elif args.task == 'train':
        train(args.now, args.latentdim, args.intdim1, args.intdim2, args.w_sent, args.mode, epochs=args.epochs, w_l=w_l)
    else:
        raise RuntimeError('Task not recognized')
