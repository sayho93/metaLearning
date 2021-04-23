import numpy as np
import sys
import tensorflow as tf
from functools import partial
from util import *
from convLayer import *


class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4, num_filters=32,
                 k_shot=5, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func = partial(cross_entropy_loss, k_shot=k_shot)
        self.dim_hidden = num_filters
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates

        # 순서대로 training data , task data
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]] * num_inner_updates
        losses_ts_post = [[]] * num_inner_updates
        accuracies_ts = [[]] * num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)

        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.conv_layers.conv_weights.keys():
                self.inner_update_lr_dict[key] = [
                    tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in
                    range(num_inner_updates)]

    # @tf.function
    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, reuse=True,
                            meta_batch_size=25, num_inner_updates=1):
            """
            Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
                Args:
                    inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                        labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                        labels used for evaluating the model after inner updates.
                        Should be shapes:
                            input_tr: [N*K, 784]
                            input_ts: [N*K, 784]
                            label_tr: [N*K, N]
                            label_ts: [N*K, N]
                Returns:
                    task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            # TODO 중요 task sepcific parameter
            weights = self.conv_layers.conv_weights

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None  # variable

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []  # list

            #######################################################################################
            # perform num_inner_updates to get modified weights
            # modified weights should be used to evaluate performance
            # Note that at each inner update, always use input_tr and label_tr for calculating gradients
            # and use input_ts and labels for evaluating performance

            # HINTS: You will need to use tf.GradientTape().
            # Read through the tf.GradientTape() documentation to see how 'persistent' should be set.
            # Here is some documentation that may be useful:
            # https://www.tensorflow.org/guide/advanced_autodiff#higher-order_gradients
            # https://www.tensorflow.org/api_docs/python/tf/GradientTape
            for i in range(num_inner_updates):
                with tf.GradientTape(persistent=True) as inner_tape:
                    # calcuating meta_train loss with training data [input_tr, label_tr]
                    task_output_tr_pre = self.conv_layers(inp=input_tr, weights=weights)
                    task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
                    # end of with block

                    #############################
                    ####      Question       ####
                    #### YOUR CODE GOES HERE ####

                    # 1) calculating gradients for inner task with testing data
                    fast_weights = weights.copy()
                    preds_tr = self.conv_layers(input_tr, weights)
                    loss = self.loss_func(preds_tr, label_tr)
                    gradients = inner_tape.gradient(loss, weights)
                    acc_tr = accuracy(label_tr, preds_tr)
                    if task_output_tr_pre is None:
                        task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = preds_tr, loss, acc_tr
                    # 2) learning fast wegiths with inner task gradients (meta parameter "self.conv_layers.conv_weights" shouldn't be updated!)
                    try:
                        for key_w, key_grads in zip(weights, gradients):
                            fast_weights[key_w] = weights[key_w] - tf.math.multiply(gradients[key_grads],
                                                                                    self.inner_update_lr_dict[
                                                                                        key_grads])
                    except AttributeError:
                        for key_w, key_grads in zip(weights, gradients):
                            fast_weights[key_w] = weights[key_w] - gradients[key_grads] * self.inner_update_lr

                    # preds_ts = self.conv_layers(input_ts, fast_weights)
                    # loss_ts = self.loss_func(preds_ts, label_ts)
                    # acc_ts = accuracy(label_ts, preds_ts)

                # calcuating adaptation loss(task_outputs_ts, task_losses_ts) with testing data [input_ts, label_ts]
                task_outputs_ts.append(self.conv_layers(input_ts, fast_weights))
                task_losses_ts.append(self.loss_func(task_outputs_ts[-1], label_ts))

                #############################

            #######################################################################################

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1),
                                            tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

            for j in range(num_inner_updates):
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1),
                                                   tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

            # Task Output
            ## task_output_tr_pre    -> tf.float32
            ## task_outputs_ts       -> [tf.float32]*num_inner_updates
            ## task_loss_tr_pre      -> tf.float32
            ## task_losses_ts        -> [tf.float32]*num_inner_updates
            ## task_accuracy_tr_pre  -> tf.float32
            ## task_accuracies_ts    -> [tf.float32]*num_inner_updates

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre,
                           task_accuracies_ts]

            return task_output

        input_tr, input_ts, label_tr, label_ts = inp  # meta train, mata validation, meta test

        # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
        unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                                 False,
                                 meta_batch_size,
                                 num_inner_updates)
        out_dtype = [tf.float32, [tf.float32] * num_inner_updates, tf.float32, [tf.float32] * num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32] * num_inner_updates])

        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size,
                                          num_inner_updates=num_inner_updates)

        result = tf.map_fn(task_inner_loop_partial,
                           elems=(input_tr, input_ts, label_tr, label_ts),
                           dtype=out_dtype,
                           parallel_iterations=meta_batch_size)

        return result
