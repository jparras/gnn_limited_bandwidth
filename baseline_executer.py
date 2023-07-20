import numpy as np
import pickle
import argparse
from tqdm import tqdm
from common import load_data_balanced, load_data_unbalanced, generate_model


def argsparser():
    parser = argparse.ArgumentParser("Baseline for distributed MNIST")
    parser.add_argument('--now', help='Number of workers', type=int, default=1)
    parser.add_argument('--mode', help='Mode of the dataset', type=str, default='balanced')
    parser.add_argument('--epochs', help='Training epochs', type=int, default=100)
    parser.add_argument('--w_sent', help='Weights sent (for pruning)', type=int, default=50000)
    return parser.parse_args()


def main(now, mode, epochs=100, alpha=0.5, w_sent=0):
    import os.path as path
    batch_size = 2048
    num_classes = 10
    fname = 'results_baseline_' + str(now) + '_' + str(mode) + '.pickle'
    if not path.isfile(fname):
        print('Processing for ', now, ' workers, ', mode, ' mode')
        # First, set memory limit dynamically
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        print('Loading datasets...')
        if mode == 'balanced':
            load_data = load_data_balanced
        elif mode == 'unbalanced':
            load_data = load_data_unbalanced
        else:
            raise RuntimeError('Mode not recognized')
        x_train, y_train, x_test, y_test, x_test_global, y_test_global, input_shape = load_data(now, num_classes)
        models = [generate_model(input_shape, num_classes) for _ in range(now)]
        if now > 1:  # Set all agents with same weights
            weights = models[0].get_weights()
            for i in range(now):
                models[i].set_weights(weights)

        print('TRAINING FOR ', now, ' WORKERS')
        loss = np.zeros((epochs, now))
        accuracy = np.zeros((epochs, now))
        loss_global = np.zeros((epochs, now))
        accuracy_global = np.zeros((epochs, now))
        for e in tqdm(range(epochs)):
            # Train each model one step on its data
            for w in range(now):
                # print('Training ', w, ' of ', n_of_workers)
                models[w].fit(x_train[w], y_train[w],
                              batch_size=batch_size,
                              epochs=1,
                              verbose=0,
                              validation_data=(x_test[w], y_test[w]))
            if now > 1:
                # Update the weights: each agent updates another
                w_vector = []
                for w in range(now):
                    w_vector.append(models[w].get_weights())
                for w in range(now):
                    # print('Updating weights ', w, ' of ', n_of_workers)
                    rec_w = (w + 1) % now  # Index of the worker that receives the update
                    weights_sender = w_vector[w]
                    if w_sent > 0:
                        # Prune!
                        weight_vector = np.concatenate([a.flatten() for a in weights_sender])
                        threshold = np.sort(np.square(weight_vector))[-w_sent]
                        for ida in weights_sender:  # For each layer
                            ida[np.square(ida) < threshold] = 0
                    models[rec_w].set_weights([(1 - alpha) * w_vector[rec_w][i] + alpha * weights_sender[i] for i in
                                               range(len(weights_sender))])

            # Obtain the score
            for w in range(now):
                # print('Obtaining the score ', w, ' of ', n_of_workers)
                score = models[w].evaluate(x_test[w], y_test[w], verbose=0)
                score_global = models[w].evaluate(x_test_global, y_test_global, verbose=0)
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
                'alpha': alpha,
                'mode': mode}
        with open(fname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Results for ', now, ' workers, ', mode, ' mode obtained')
    else:
        print('Results for ', now, ' workers, ', mode, ' mode already exist')


if __name__ == '__main__':

    args = argsparser()
    main(args.now, args.mode, w_sent=args.w_sent, epochs=args.epochs)
