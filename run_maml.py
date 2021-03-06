import csv
import numpy as np
import pickle
import random
import tensorflow as tf
from models.maml import *
from load_data import *

def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def meta_train_fn(model, exp_string, data_generator,
                  n_way=5, meta_train_iterations=15000, meta_batch_size=25,
                  log=True, logdir='./log/model', k_shot=1, num_inner_updates=1, meta_lr=0.001):
    tf.debugging.set_log_device_placement(True)

    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    pre_accuracies, post_accuracies = [], []

    num_classes = data_generator.num_classes
    img_size = data_generator.img_size

    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    for itr in range(meta_train_iterations):

        # sample a batch of training data and partition into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same number of examples.

        all_image_batches, all_label_batches = data_generator.sample_batch(batch_type='meta_train',
                                                                           batch_size=meta_batch_size, shuffle=False,
                                                                           swap=False)

        input_tr, input_ts = all_image_batches[:, :, :k_shot, :], all_image_batches[:, :, k_shot:, :]
        label_tr, label_ts = all_label_batches[:, :, :k_shot, :], all_label_batches[:, :, k_shot:, :]

        # reshape input tensor
        input_tr = tf.reshape(input_tr, [-1, n_way * k_shot, img_size[0] * img_size[0]])
        input_ts = tf.reshape(input_ts, [-1, n_way * k_shot, img_size[0] * img_size[0]])
        label_tr = tf.reshape(label_tr, [-1, n_way * k_shot, n_way])
        label_ts = tf.reshape(label_ts, [-1, n_way * k_shot, n_way])

        inp = (input_tr, input_ts, label_tr, label_ts)

        result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size,
                                  num_inner_updates=num_inner_updates)

        if itr % SUMMARY_INTERVAL == 0:
            pre_accuracies.append(result[-2])
            post_accuracies.append(result[-1][-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (
            itr, np.mean(pre_accuracies), np.mean(post_accuracies))
            print(print_str)
            pre_accuracies, post_accuracies = [], []

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            # sample a batch of validation data and partition it into
            # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
            # NOTE: The code assumes that the support and query sets have the same number of examples.

            all_image_batches, all_label_batches = data_generator.sample_batch(batch_type='meta_val',
                                                                               batch_size=meta_batch_size,
                                                                               shuffle=False, swap=False)

            input_tr, input_ts = all_image_batches[:, :, :k_shot, :], all_image_batches[:, :, k_shot:, :]
            label_tr, label_ts = all_label_batches[:, :, :k_shot, :], all_label_batches[:, :, k_shot:, :]

            # reshape input tensor
            input_tr = tf.reshape(input_tr, [-1, n_way * k_shot, img_size[0] * img_size[0]])
            input_ts = tf.reshape(input_ts, [-1, n_way * k_shot, img_size[0] * img_size[0]])
            label_tr = tf.reshape(label_tr, [-1, n_way * k_shot, n_way])
            label_ts = tf.reshape(label_ts, [-1, n_way * k_shot, n_way])

            inp = (input_tr, input_ts, label_tr, label_ts)
            result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            print(
                'Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (
                result[-2], result[-1][-1]))

            model_file = logdir + '/' + exp_string + '/model' + str(itr)
            print("Saving to ", model_file)
            model.save_weights(model_file)

    model_file = logdir + '/' + exp_string + '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)


# calculated for omniglot
NUM_META_TEST_POINTS = 600


def meta_test_fn(model, data_generator, n_way=5,
                 meta_batch_size=25, k_shot=1,
                 num_inner_updates=1):
    num_classes = data_generator.num_classes
    img_size = data_generator.img_size

    np.random.seed(1)
    random.seed(1)

    meta_test_accuracies = []

    for _ in range(NUM_META_TEST_POINTS):
        # sample a batch of test data and partition it into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same number of examples.

        all_image_batches, all_label_batches = data_generator.sample_batch(batch_type='meta_test',
                                                                           batch_size=meta_batch_size, shuffle=False,
                                                                           swap=False)

        input_tr, input_ts = all_image_batches[:, :, :k_shot, :], all_image_batches[:, :, k_shot:, :]
        label_tr, label_ts = all_label_batches[:, :, :k_shot, :], all_label_batches[:, :, k_shot:, :]

        # reshape input tensor
        input_tr = tf.reshape(input_tr, [-1, n_way * k_shot, img_size[0] * img_size[0]])
        input_ts = tf.reshape(input_ts, [-1, n_way * k_shot, img_size[0] * img_size[0]])
        label_tr = tf.reshape(label_tr, [-1, n_way * k_shot, n_way])
        label_ts = tf.reshape(label_ts, [-1, n_way * k_shot, n_way])

        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        meta_test_accuracies.append(result[-1][-1])

    meta_test_accuracies = np.array(meta_test_accuracies)
    means = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    ci95 = 1.96 * stds / np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))


def run_maml(n_way=5, k_shot=1, meta_batch_size=25, meta_lr=0.001,
             inner_update_lr=0.4, num_filters=32, num_inner_updates=1,
             learn_inner_update_lr=False,
             resume=False, resume_itr=0, log=True, logdir='./log/model',
             data_path='./omniglot_resized', meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1):
    # call data_generator and get data with k_shot*2 samples per class
    data_generator = DataGenerator(n_way, k_shot * 2, n_way, k_shot * 2, config={'data_folder': data_path})

    # set up MAML model
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    model = MAML(dim_input,
                 dim_output,
                 num_inner_updates=num_inner_updates,
                 inner_update_lr=inner_update_lr,
                 k_shot=k_shot,
                 num_filters=num_filters,
                 learn_inner_update_lr=learn_inner_update_lr)

    if meta_train_k_shot == -1:
        meta_train_k_shot = k_shot
    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'cls_' + str(n_way) + '.mbs_' + str(meta_batch_size) + '.k_shot_' + str(
        meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(
        meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

    if meta_train:
        meta_train_fn(model, exp_string, data_generator,
                      n_way, meta_train_iterations, meta_batch_size, log, logdir,
                      k_shot, num_inner_updates, meta_lr)
    else:
        meta_batch_size = 1

        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)