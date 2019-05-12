#  ***********************************************************************
#
#   This file defines some common LSTM network architectures
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************

import collections
import numpy as np
import warnings
import tensorflow as tf
import twodlearn as tdl
from twodlearn import common
import twodlearn.feedforward as tdlf
from twodlearn.feedforward import (AffineLayer, DenseLayer, NetConf)
# TODO: change the saver to be defined layer by layer


class LstmLayer:
    def __init__(self, FinalOut, inputs_list,
                 last_output, last_state, labels_list,
                 saved_output, saved_state,
                 assign_saved_out_state, reset_saved_out_state,
                 error_per_sample):
        # in the case of classification, this are the logits, i.e. the output from the final linear transformation
        self.y = FinalOut
        self.inputs_list = inputs_list  # list of tensor placeholders for the inputs
        self.labels_list = labels_list  # list of tensor placeholders for the labels

        # list of tensor variables for the last output of the unrolling
        self.last_output = last_output
        # list of tensor variables for the last state of the unrolling
        self.last_state = last_state

        self.saved_output = saved_output  # list of tensor placeholders for the inputs
        self.saved_state = saved_state    # list of tensor placeholders for the inputs

        # assigns last state and output to saved_state and saved_output
        self.assign_saved_out_state = assign_saved_out_state

        ''''''
        self.reset_saved_out_state = tf.group(
            *reset_saved_out_state)  # resets saved_state and saved_output
        self.error_per_sample = error_per_sample


class SimpleLstmCell(object):
    ''' Single lstm cell

    Attributes:
        n_inputs: number of inputs
        n_nodes: nuber of nodes
        afunction: activation function, for the moment it could be tanh and ReLU
        name: name used in all TensorFlow variables' names

        saver_dict: saver for the parameters used by the layer
    '''

    def __init__(self, n_inputs, n_nodes, afunction='tanh', name=''):

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.afunction = afunction
        self.name = name

        # 1. Define Trainable Parameters:
        # Input gate: input, previous output, and bias.
        self.ix = tf.Variable(tf.truncated_normal(
            [n_inputs, self.n_nodes], -0.1, 0.1), name=('w_ix' + name))
        self.im = tf.Variable(tf.truncated_normal(
            [self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_im' + name))
        self.ib = tf.Variable(
            tf.zeros([1, self.n_nodes]), name=('w_ib' + name))
        # Forget gate: input, previous output, and bias.
        self.fx = tf.Variable(tf.truncated_normal(
            [n_inputs, self.n_nodes], -0.1, 0.1), name=('w_fx' + name))
        self.fm = tf.Variable(tf.truncated_normal(
            [self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_fm' + name))
        self.fb = tf.Variable(
            tf.zeros([1, self.n_nodes]), name=('w_fb' + name))
        # Memory cell: input, state and bias.
        self.cx = tf.Variable(tf.truncated_normal(
            [n_inputs, self.n_nodes], -0.1, 0.1), name=('w_cx' + name))
        self.cm = tf.Variable(tf.truncated_normal(
            [self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_cm' + name))
        self.cb = tf.Variable(
            tf.zeros([1, self.n_nodes]), name=('w_cb' + name))
        # Output gate: input, previous output, and bias.
        self.ox = tf.Variable(tf.truncated_normal(
            [n_inputs, self.n_nodes], -0.1, 0.1), name=('w_ox' + name))
        self.om = tf.Variable(tf.truncated_normal(
            [self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_om' + name))
        self.ob = tf.Variable(
            tf.zeros([1, self.n_nodes]), name=('w_ob' + name))

        # Define the saver:
        self.saver_dict = dict()
        self.saver_dict['w_ix' + name] = self.ix
        self.saver_dict['w_im' + name] = self.im
        self.saver_dict['b_i' + name] = self.ib
        self.saver_dict['w_fx' + name] = self.fx
        self.saver_dict['w_fm' + name] = self.fm
        self.saver_dict['b_f' + name] = self.fb
        self.saver_dict['w_cx' + name] = self.cx
        self.saver_dict['w_cm' + name] = self.cm
        self.saver_dict['b_c' + name] = self.cb
        self.saver_dict['w_ox' + name] = self.ox
        self.saver_dict['w_om' + name] = self.om
        self.saver_dict['b_o' + name] = self.ob

    # Definition of the cell computation.
    def evaluate(self, i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""

        if self.afunction == 'tanh':
            input_gate = tf.sigmoid(
                tf.matmul(i, self.ix) + tf.matmul(o, self.im) + self.ib)
            forget_gate = tf.sigmoid(
                tf.matmul(i, self.fx) + tf.matmul(o, self.fm) + self.fb)
            update = tf.matmul(i, self.cx) + tf.matmul(o, self.cm) + self.cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(
                tf.matmul(i, self.ox) + tf.matmul(o, self.om) + self.ob)
            return output_gate * tf.tanh(state), state

        if self.afunction == 'ReLU':
            input_gate = tf.sigmoid(
                tf.matmul(i, self.ix) + tf.matmul(o, self.im) + self.ib)
            forget_gate = tf.sigmoid(
                tf.matmul(i, self.fx) + tf.matmul(o, self.fm) + self.fb)
            update = tf.matmul(i, self.cx) + tf.matmul(o, self.cm) + self.cb
            state = forget_gate * state + input_gate * tf.nn.relu(update)
            output_gate = tf.sigmoid(
                tf.matmul(i, self.ox) + tf.matmul(o, self.om) + self.ob)
            return output_gate * tf.nn.relu(state), state


class AlexLstmCell(object):
    ''' Single lstm cell defined as in: "Generating Sequences with Recurrent Neural Networks", Alex Graves, 2014

    Attributes:
        n_inputs: number of inputs
        n_nodes: number of nodes
        afunction: activation function, for the moment it could be tanh and ReLU, TODO: change it to any function
        name: name used in all TensorFlow variables' names
    '''

    def __init__(self, n_inputs, n_nodes, afunction='tanh', name=''):

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.afunction = afunction
        self.name = name

        # 1. Define Trainable Parameters:
        # Note: w_ci, w_cf, wc_o are a diagonal matrices, therefore we only reserve space for the diagonal, and the product with the cell state is done as a element-by-element product instead of a matrix product
        # Input gate: input, previous state, previous hidden output and bias.
        self.w_xi = tf.Variable(tf.truncated_normal(
            [n_inputs, n_nodes], -0.1, 0.1), name=('w_xi' + name))
        self.w_hi = tf.Variable(tf.truncated_normal(
            [n_nodes, n_nodes], -0.1, 0.1), name=('w_hi' + name))
        self.w_ci = tf.Variable(tf.truncated_normal(
            [n_nodes], -0.1, 0.1), name=('w_ci' + name))
        self.b_i = tf.Variable(tf.zeros([1, n_nodes]), name=('b_i' + name))
        # Forget gate: input, previous state, and bias.
        self.w_xf = tf.Variable(tf.truncated_normal(
            [n_inputs, n_nodes], -0.1, 0.1), name=('w_xf' + name))
        self.w_hf = tf.Variable(tf.truncated_normal(
            [n_nodes, n_nodes], -0.1, 0.1), name=('w_hi' + name))
        self.w_cf = tf.Variable(tf.truncated_normal(
            [n_nodes], -0.1, 0.1), name=('w_cf' + name))
        self.b_f = tf.Variable(tf.zeros([1, n_nodes]), name=('b_f' + name))
        # Memory cell: input, state and bias.
        self.w_xc = tf.Variable(tf.truncated_normal(
            [n_inputs, n_nodes], -0.1, 0.1), name=('w_xc' + name))
        self.w_hc = tf.Variable(tf.truncated_normal(
            [n_nodes, n_nodes], -0.1, 0.1), name=('w_hc' + name))
        self.b_c = tf.Variable(tf.zeros([1, n_nodes]), name=('b_c' + name))
        # Output gate: input, previous output, and bias.
        self.w_xo = tf.Variable(tf.truncated_normal(
            [n_inputs, n_nodes], -0.1, 0.1), name=('w_xo' + name))
        self.w_ho = tf.Variable(tf.truncated_normal(
            [n_nodes, n_nodes], -0.1, 0.1), name=('w_ho' + name))
        self.w_co = tf.Variable(tf.truncated_normal(
            [n_nodes], -0.1, 0.1), name=('w_co' + name))
        self.b_o = tf.Variable(tf.zeros([1, n_nodes]), name=('b_o' + name))

        # Define the saver:
        self.saver_dict = dict()
        self.saver_dict['w_xi' + name] = self.w_xi
        self.saver_dict['w_hi' + name] = self.w_hi
        self.saver_dict['w_ci' + name] = self.w_ci
        self.saver_dict['b_i' + name] = self.b_i
        self.saver_dict['w_xf' + name] = self.w_xf
        self.saver_dict['w_hf' + name] = self.w_hf
        self.saver_dict['w_cf' + name] = self.w_cf
        self.saver_dict['b_f' + name] = self.b_f
        self.saver_dict['w_xc' + name] = self.w_xc
        self.saver_dict['w_hc' + name] = self.w_hc
        self.saver_dict['b_c' + name] = self.b_c
        self.saver_dict['w_xo' + name] = self.w_xo
        self.saver_dict['w_ho' + name] = self.w_ho
        self.saver_dict['w_co' + name] = self.w_co
        self.saver_dict['b_o' + name] = self.b_o

    def evaluate(self, i, h, state):
        """Create a LSTM cell.
        i: inputs, on the paper (Alex Graves) the notation is x_t
        h: hidden output on the previous time step, on the paper is h_t-1
        state: state on the previous time step, in the paper is c_t-1
        """

        if self.afunction == 'tanh':
            i_t = tf.sigmoid(tf.matmul(i, self.w_xi) + tf.matmul(h,
                                                                 self.w_hi) + state * self.w_ci + self.b_i)  # input gate
            f_t = tf.sigmoid(tf.matmul(i, self.w_xf) + tf.matmul(h,
                                                                 self.w_hf) + state * self.w_cf + self.b_f)  # forget gate
            update = tf.matmul(i, self.w_xc) + tf.matmul(h,
                                                         self.w_hc) + self.b_c  # state update
            state = f_t * state + i_t * tf.tanh(update)
            o_t = tf.sigmoid(tf.matmul(i, self.w_xo) + tf.matmul(h,
                                                                 self.w_ho) + state * self.w_co + self.b_o)
            return o_t * tf.tanh(state), state


class AlexLstmCellOptimized(object):
    ''' Single lstm cell defined as in: "Generating Sequences with Recurrent Neural Networks", Alex Graves, 2014

    Attributes:
        n_inputs: number of inputs
        n_nodes: number of nodes
        afunction: activation function, for the moment it could be tanh and ReLU, TODO: change it to any function
        name: name used in all TensorFlow variables' names
    '''

    def __init__(self, n_inputs, n_nodes, afunction='tanh', name=''):

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.afunction = afunction
        self.name = name

        # 1. Define Trainable Parameters:
        # Note: w_ci, w_cf, wc_o are a diagonal matrices, therefore we only reserve space for the diagonal, and the product with the cell state is done as a element-by-element product instead of a matrix product

        self.w_x = tf.Variable(tf.truncated_normal(
            [n_inputs, 4 * n_nodes], -0.1, 0.1), name=('w_xi' + name))
        self.w_h = tf.Variable(tf.truncated_normal(
            [n_nodes, 4 * n_nodes], -0.1, 0.1), name=('w_hi' + name))
        # self.w_cprev = tf.Variable(tf.truncated_normal([n_nodes,2], -0.1, 0.1), name=('w_ci'+name))
        # self.w_c = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_ci'+name))

        # Input gate
        self.w_ci = tf.Variable(tf.truncated_normal(
            [n_nodes], -0.1, 0.1), name=('w_ci' + name))
        self.b_i = tf.Variable(tf.zeros([1, n_nodes]), name=('b_i' + name))
        # Forget gate: input, previous state, and bias.
        self.w_cf = tf.Variable(tf.truncated_normal(
            [n_nodes], -0.1, 0.1), name=('w_cf' + name))
        self.b_f = tf.Variable(tf.zeros([1, n_nodes]), name=('b_f' + name))
        # Memory cell: input, state and bias.
        self.b_c = tf.Variable(tf.zeros([1, n_nodes]), name=('b_c' + name))
        # Output gate: input, previous output, and bias.
        self.w_co = tf.Variable(tf.truncated_normal(
            [n_nodes], -0.1, 0.1), name=('w_co' + name))
        self.b_o = tf.Variable(tf.zeros([1, n_nodes]), name=('b_o' + name))

        # Define the saver:
        self.saver_dict = dict()
        self.saver_dict['w_x' + name] = self.w_x
        self.saver_dict['w_h' + name] = self.w_h

        self.saver_dict['w_ci' + name] = self.w_ci
        self.saver_dict['w_cf' + name] = self.w_cf
        self.saver_dict['w_co' + name] = self.w_co

        self.saver_dict['b_i' + name] = self.b_i
        self.saver_dict['b_f' + name] = self.b_f
        self.saver_dict['b_c' + name] = self.b_c
        self.saver_dict['b_o' + name] = self.b_o

    def evaluate(self, i, h, state):
        """Create a LSTM cell.
        i: inputs, on the paper (Alex Graves) the notation is x_t
        h: hidden output on the previous time step, on the paper is h_t-1
        state: state on the previous time step, in the paper is c_t-1
        """

        if self.afunction == 'tanh':
            # x transformation
            aux_x = tf.matmul(i, self.w_x)
            ix, fx, cx, ox = tf.split(aux_x, 4, axis=1)

            # h transformation
            aux_h = tf.matmul(h, self.w_h)
            ih, fh, ch, oh = tf.split(aux_h, 4, axis=1)

            # calculate gate values and output
            i_t = tf.sigmoid(ix + ih + state * self.w_ci +
                             self.b_i)  # input gate
            f_t = tf.sigmoid(fx + fh + state * self.w_cf +
                             self.b_f)  # forget gate
            update = cx + ch + self.b_c  # state update
            state = f_t * state + i_t * tf.tanh(update)
            o_t = tf.sigmoid(ox + oh + state * self.w_co + self.b_o)
            return o_t * tf.tanh(state), state


class LstmNet(object):
    ''' network with multiple lstm cells:

    inputs_list -> [lstm_cell_1] -> [lstm_cell_2] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list

    Attributes:
        n_inputs: number of inputs
        n_nodes: list with the number of nodes on each layer
        n_outputs: number of outputs
        n_extra: list with the number of extra inputs per layer (including first layer)
        n_layers: number of layers
        name: name used in all TensorFlow variables' names

        w: weights for final linear transformation
        b: bias for final linear transformation

    '''

    def __init__(self, n_inputs, n_nodes, n_outputs, n_extra=0,
                 afunction='tanh',
                 LstmCell=SimpleLstmCell,
                 OutLayer=AffineLayer,
                 name=''):
        ''' Define the variables that will represent the learning parameters of the network'''
        # setup inputs and outputs of the network
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.name = name

        self.n_layers = len(n_nodes)

        # check n_extra values
        if isinstance(n_extra, list):
            self.n_extra = n_extra
            if len(self.n_extra) != self.n_layers + 1:
                raise ValueError(
                    'n_extra list must have a length equal to the number of hidden layers plus the output layer')

            if (self.n_extra[0] != 0):
                raise ValueError(
                    'n_extra[0] must be zero at the moment (no extra inputs for first layer)')

        else:
            # all hidden cells plus the output layer
            self.n_extra = [n_extra for i in range(n_layers + 1)]

        # Create each one of the cells for each layer
        self.cell_list = list()

        self.cell_list.append(
            LstmCell(n_inputs + self.n_extra[0], n_nodes[0], afunction, name=name + '_L0'))

        for l in range(1, self.n_layers):
            self.cell_list.append(LstmCell(
                n_nodes[l - 1] + self.n_extra[l], n_nodes[l], afunction, name=name + '_L' + str(l)))

        # Final output weights and biases.
        if OutLayer is not None:
            self.n_outputs = n_outputs
            self.out_layer = OutLayer(
                n_nodes[-1] + self.n_extra[-1], n_outputs)
        else:
            self.out_layer = None

        # Saver
        saver_dict = dict()
        for l in range(self.n_layers):
            saver_dict.update(self.cell_list[l].saver_dict)

        if self.out_layer is not None:
            saver_dict.update(self.out_layer.saver_dict)

        self.saver = tf.train.Saver(saver_dict)

    def get_extra_inputs(self, i, h_list, state_list):
        ''' Gets extra inputs for current layer
        This function should be overwrited by the user if he wants to introduce aditional inputs to the neural network
        i: input to the network
        h_list: list with the hidden outputs of each layer up to the current layer
        state_list: list with the value of the internal state for each cell
        '''
        return None

    def evaluate_final_output(self, outputs_list, inputs_list, h_list):
        ''' Calculates the final output of the neural network, usually it is just a linear transformation

        outputs_list: list with the outputs from the last lstm cell
        inputs_list: list of inputs to the network
        h_list: list with all hidden outputs from all the cells, Note: h_list includes outputs_list
        '''
        return self.out_layer.evaluate(tf.concat(0, outputs_list))

    def lstm_net_step(self, i, o_prev, state_prev, drop_prob_list):
        """  Calculates the entire set of next states and outputs for input i

        Each layer has its state and output, therefore state and output are lists with
        the state and output for each layer
        i: input to the network
        o_prev: list of hidden outputs in the previous time step
        state_prev: list with the states in the previous time step
        drop_prob_list: list with the dropout placeholders
        """
        out_list = list()
        state_list = list()

        # first layer
        if self.n_extra[0] > 0:
            # handle extra inputs
            i = tf.concat(
                1, [i, self.get_extra_inputs(i, out_list, state_list)])

        if drop_prob_list[0] is not None:
            output, state = self.cell_list[0].evaluate(
                tf.nn.dropout(i, drop_prob_list[0]), o_prev[0], state_prev[0])
        else:
            output, state = self.cell_list[0].evaluate(
                i, o_prev[0], state_prev[0])
        out_list.append(output)
        state_list.append(state)

        # all following layers
        for l in range(1, self.n_layers):
            if self.n_extra[l] > 0:
                # handle extra inputs
                output = tf.concat(
                    1, [output, self.get_extra_inputs(i, out_list, state_list)])
            if drop_prob_list[l] is not None:
                output, state = self.cell_list[l].evaluate(tf.nn.dropout(
                    output, drop_prob_list[l]), o_prev[l], state_prev[l])
            else:
                output, state = self.cell_list[l].evaluate(
                    output, o_prev[l], state_prev[l])

            out_list.append(output)
            state_list.append(state)

        return out_list, state_list

    # Definition of the cell computation.
    def unrolling_setup(self, batch_size, num_unrollings,
                        inputs_list=None,
                        labels_list=None,
                        drop_prob_list=None,
                        saved_output=None, saved_state=None,
                        deps_list=None,
                        calculate_loss=True,
                        reset_between_unrollings=False,
                        name=''
                        ):
        """Unrolls the lstm network.

        Creates an unrolling of the network:

        inputs_list -> [lstm_cell_1] -> [lstm_cell_1] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list
                             ^                ^           ^           ^
                             |                |           |           |
        inputs_list -> [lstm_cell_1] -> [lstm_cell_1] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list
                             ^                ^           ^           ^
                             |                |           |           |
        inputs_list -> [lstm_cell_1] -> [lstm_cell_1] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list

        Args:
            batch_size:
            num_unrollings:
            inputs_list:
            labels_list: user can optionally provide its own labels placeholders
            drop_prob_list:
            saved_output: user can optionally provide its own initialization for the output tensor
            saved_state: user can optionally provide its own initialization for the state tensor
            deps_list: list of dependencies to be runned before calculating final output y and loss
            reset_between_unrollings: reset the saved state and output between unrolling calls

        Returns:
            A class that defines the important variables for using the unrolling:
        """

        # 1. Saved output and state from previous unrollings
        create_feedback = False
        if saved_output is None:
            saved_output = list()
            saved_state = list()
            for l in range(self.n_layers):
                saved_output.append(tf.Variable(tf.zeros([batch_size, self.n_nodes[l]]), trainable=False,
                                                name='saved_output_L' + str(l) + self.name + name))
                saved_state.append(tf.Variable(tf.zeros([batch_size, self.n_nodes[l]]), trainable=False,
                                               name='saved_state_L' + str(l) + self.name + name))
            create_feedback = True

        # 2. Input data.
        if inputs_list is None:
            inputs_list = list()
            for iaux in range(num_unrollings):
                inputs_list.append(tf.placeholder(tf.float32, shape=[batch_size, self.n_inputs],
                                                  name='inputs_list' + str(iaux) + self.name + name))

        # 3. Unrolled LSTM loop.
        outputs_list = list()  # list with the hidden output of the last cell
        h_list = list()        # list with all hidden outputs of the network, including outputs_list
        output = saved_output
        state = saved_state
        # print(state[1]) DELETE
        for i in inputs_list:
            # output, state = self.lstm_cell(i, output, state) # we introduce dropout here
            output, state = self.lstm_net_step(
                i, output, state, drop_prob_list)

            h_list.append(output)

            if drop_prob_list[-1] is not None:
                outputs_list.append(tf.nn.dropout(
                    output[-1], drop_prob_list[-1]))
            else:
                outputs_list.append(output[-1])

        # Create a list that assigns state and output to saved state and output
        if create_feedback:
            assign_saved_out_state = list()
            for l in range(self.n_layers):
                assign_saved_out_state.append(
                    saved_output[l].assign(output[l]))
                assign_saved_out_state.append(saved_state[l].assign(state[l]))
        else:
            assign_saved_out_state = None

        # Create a list to reset saved_state and saved_output
        reset_output_state = list()
        for l in range(self.n_layers):
            reset_output_state.append(saved_output[l].assign(
                tf.zeros([batch_size, self.n_nodes[l]])))
            reset_output_state.append(saved_state[l].assign(
                tf.zeros([batch_size, self.n_nodes[l]])))

        # 4. Final output
        if calculate_loss and (self.out_layer is not None):
            # First create a place holder for labels to be able to calculate the loss
            if labels_list is None:
                labels_list = list()
                for _ in range(num_unrollings):
                    labels_list.append(tf.placeholder(
                        tf.float32, shape=[batch_size, self.n_outputs]))

            if deps_list is None:
                if reset_between_unrollings:
                    with tf.control_dependencies(reset_output_state):
                        # y = tf.nn.xw_plus_b(tf.concat(0, outputs_list), self.w, self.b)
                        y = self.evaluate_final_output(
                            outputs_list, inputs_list, h_list)
                        # For regression:
                        # error_per_sample= y - tf.concat(0, labels_list)
                        # loss = tf.nn.l2_loss( error_per_sample )
                        # For clasification:
                        error_per_sample = tf.nn.softmax_cross_entropy_with_logits(
                            y, tf.concat(0, labels_list))
                        loss = tf.reduce_mean(error_per_sample)
                else:
                    with tf.control_dependencies(assign_saved_out_state):
                        # y = tf.nn.xw_plus_b(tf.concat(0, outputs_list), self.w, self.b)
                        y = self.evaluate_final_output(
                            outputs_list, inputs_list, h_list)
                        # For regression:
                        # error_per_sample= y - tf.concat(0, labels_list)
                        # loss = tf.nn.l2_loss( error_per_sample )
                        # For clasification:
                        error_per_sample = tf.nn.softmax_cross_entropy_with_logits(
                            y, tf.concat(0, labels_list))
                        loss = tf.reduce_mean(error_per_sample)
            else:
                with tf.control_dependencies(deps_list):
                    # y = tf.nn.xw_plus_b(tf.concat(0, outputs_list), self.w, self.b)
                    y = self.evaluate_final_output(
                        outputs_list, inputs_list, h_list)
                    # For regression:
                    # error_per_sample= y - tf.concat(0, labels_list)
                    # loss = tf.nn.l2_loss( error_per_sample )
                    # For clasification:
                    error_per_sample = tf.nn.softmax_cross_entropy_with_logits(
                        y, tf.concat(0, labels_list))
                    loss = tf.reduce_mean(error_per_sample)

            # 5. Return
            return LstmLayer(y, inputs_list,
                             output, state, labels_list,
                             saved_output, saved_state,
                             assign_saved_out_state, reset_output_state,
                             error_per_sample), loss

        else:
            # if deps_list is None:
            return LstmLayer(None, inputs_list,
                             output, state, None,
                             saved_output, saved_state,
                             assign_saved_out_state, reset_output_state,
                             None), None


class AlexLstmNet(LstmNet):

    def __init__(self, n_inputs, n_nodes, n_outputs, n_extra=0,
                 afunction='tanh',
                 LstmCell=SimpleLstmCell,
                 OutLayer=AffineLayer,
                 name=''):

        # check n_extra values
        if isinstance(n_extra, list):
            self.n_extra = n_extra
            if len(self.n_extra) != self.n_layers + 1:
                raise ValueError(
                    'n_extra list must have a length equal to the number of hidden layers plus the output layer')

            if (self.n_extra[0] != 0):
                raise ValueError(
                    'n_extra[0] must be zero at the moment (no extra inputs for first layer)')

        else:
            if len(num_nodes) > 1:
                # for all hidden cells plus the output layer:
                n_extra = [num_inputs for i in range(len(num_nodes) + 1)]
                n_extra[0] = 0
                n_extra[-1] = sum(num_nodes) - num_nodes[-1]
            else:
                n_extra = [0, 0]

        # call init form superclass LstmNet
        super().__init__(n_inputs, n_nodes, n_outputs,
                         n_extra, afunction, LstmCell, OutLayer, name)

    def get_extra_inputs(self, i, h_list, state_list):
        return i

    def evaluate_final_output(self, outputs_list, inputs_list, h_list):
        ''' Calculates the final output of the neural network, usually it is just a linear transformation

        outputs_list: list with the outputs from the last lstm cell
        inputs_list: list of inputs to the network
        h_list: list with all hidden outputs from all the cells
        '''
        ''''''
        all_hidden = list()

        for t in h_list:  # go trough each time step
            all_hidden.append(tf.concat(1, t))

        return self.out_layer.evaluate(tf.concat(0, all_hidden))


class AlexLstmNet_MemOpt(LstmNet):  # TODO

    def __init__(self, n_inputs, n_nodes, n_outputs, n_extra=0,
                 afunction='tanh',
                 LstmCell=SimpleLstmCell,
                 OutLayer=AffineLayer,
                 name=''):

        # check n_extra values
        if isinstance(n_extra, list):
            self.n_extra = n_extra
            if len(self.n_extra) != self.n_layers + 1:
                raise ValueError(
                    'n_extra list must have a length equal to the number of hidden layers plus the output layer')

            if (self.n_extra[0] != 0):
                raise ValueError(
                    'n_extra[0] must be zero at the moment (no extra inputs for first layer)')

        else:
            if len(num_nodes) > 1:
                # for all hidden cells plus the output layer:
                n_extra = [num_inputs for i in range(len(num_nodes) + 1)]
                n_extra[0] = 0
                n_extra[-1] = sum(num_nodes) - num_nodes[-1]
            else:
                n_extra = [0, 0]

        # call init form superclass LstmNet
        super().__init__(n_inputs, n_nodes, n_outputs,
                         n_extra, afunction, LstmCell, OutLayer, name)

    def get_extra_inputs(self, i, h_list, state_list):
        return i

    def evaluate_final_output(self, outputs_list, inputs_list, h_list):
        ''' Calculates the final output of the neural network, usually it is just a linear transformation

        outputs_list: list with the outputs from the last lstm cell
        inputs_list: list of inputs to the network
        h_list: list with all hidden outputs from all the cells
        '''
        ''''''
        all_hidden = list()

        for t in h_list:  # go trough each time step
            all_hidden.append(tf.concat(1, t))

        y_list = list()
        for t in all_hidden:
            y_list.append(self.out_layer.evaluate(t))

        return tf.concat(0, y_list)


class SimpleRnn(common.TdlModel):
    @common.Submodel
    def cell(self, value):
        return (value if value is not None
                else self._define_cell())

    @common.Regularizer
    def regularizer(self, *args, **kargs):
        '''Initializes the regularizer using the cell regularizer'''
        reg = (self.cell.regularizer.value
               if self.cell.regularizer.is_set
               else self.cell.regularizer.init(*args, **kargs))
        return reg

    def __init__(self, cell, name='SimpleRnn', options=None, **kargs):
        super(SimpleRnn, self).__init__(cell=cell, options=options,
                                        name=name, **kargs)

    class RnnOutput(common.TdlModel):
        @property
        def n_unrollings(self):
            return self._n_unrollings

        @property
        def x(self):
            return self._x

        @property
        def y(self):
            return self._y

        @property
        def unrolled(self):
            ''' list of unrolled networks '''
            return self._unrolled

        @common.LazzyProperty
        def reset_inputs(self):
            reset_inputs = tf.variables_initializer(
                var_list=tdl.get_trainable(self.inputs),
                name="reset_inputs")
            return reset_inputs

        @common.InputArgument
        def inputs(self, value):
            ''' Setup either tf variables or placeholders for the external
            (control)  inputs '''
            if value is not None:
                return value

            if self.options['inputs/type'] == 'placeholder':
                inputs = list()
                for k in range(self.n_unrollings):
                    inputs_k = tf.placeholder(
                        tf.float32, shape=self.options['inputs/shape'],
                        name='input_{}'.format(k))
                    inputs.append(inputs_k)
            elif self.options['inputs/type'] == 'variable':
                inputs = list()
                for k in range(self.n_unrollings):
                    inputs_k = tdl.variable(
                        tf.random_normal(shape=self.options['inputs/shape'],
                                         mean=self.options['inputs/mean'],
                                         stddev=self.options['inputs/std']),
                        name='inputs_{}'.format(k))
                    inputs.append(inputs_k)
            else:
                raise ValueError('available options for inputs/type are: '
                                 '[placeholder, variable]')
            return inputs

        @common.InputArgument
        def x0(self, value):
            if value is None:
                raise NotImplementedError('x0 is not defined, please specify '
                                          'the x0 initialization method')
            else:
                return value

        def _xt_transfer_func(self, xt, t):
            return xt

        def _ut_transfer_func(self, ut, t):
            return ut

        def _next_step(self, xt, ut, t):
            """Define the operation for computing the next step
            Args:
                xt (type): current state, in case of narx, previous window_size
                           concatenated observations.
                ut (type): current exogenous input.
                t (int): current time step.
            Returns:
                tuple(yt, xt, net): output, state and net at time t
            """
            yt = None
            xt = None
            net = None
            raise NotImplementedError('No setup defined, please specify the '
                                      '_next_step method')
            return yt, xt, net

        def _define_step_loss(self, yt, xt, ut, net_t, t, T):
            ''' Loss corresponding to step time t '''
            return (net_t.loss.value if net_t.loss.is_set
                    else net_t.loss.init())

        @common.OptionalProperty
        def loss(self, reg_alpha=None):
            ''' defines the loss function
                sum(unrolled.loss) + alpha*cell.regularizer
            '''
            step_loss = [self._define_step_loss(yt=self.y[t], xt=self.x[t],
                                                ut=self.inputs[t],
                                                net_t=self.unrolled[t],
                                                t=t, T=len(self.y))
                         for t in range(self.n_unrollings)]
            loss = tdlf.AddNLosses(step_loss)
            fit_loss = tdlf.ScaledLoss(1.0/self.n_unrollings, loss)
            if all([isinstance(l, tdlf.EmpiricalLoss) for l in loss]):
                fit_loss = tdlf.EmpiricalLossWrapper(
                    loss=fit_loss,
                    labels=[l.labels for l in loss])

            if hasattr(self.model, 'regularizer') and reg_alpha is not None:
                if self.model.regularizer.is_set:
                    loss = tdlf.EmpiricalWithRegularization(
                        empirical=fit_loss,
                        regularizer=self.model.regularizer.value,
                        alpha=reg_alpha)
                else:
                    raise ValueError('reg_alpha was specified but the model '
                                     'regularizer has not been specified')
            return loss

        def _init_options(self, options):
            default = {'inputs/mean': 0.0,
                       'inputs/std': 0.1,
                       'inputs/type': 'placeholder',
                       'inputs/shape': None
                       }
            options = common.check_defaults(options, default)
            options = super(SimpleRnn.RnnOutput, self)._init_options(options)
            return options

        def __init__(self, model, x0=None, inputs=None, n_unrollings=None,
                     options=None, name=None):
            self.model = model
            assert (n_unrollings is not None or inputs is not None),\
                'must specify either number of unrollings or inputs'
            self._n_unrollings = (len(inputs) if n_unrollings is None
                                  else n_unrollings)
            if inputs is not None:
                assert (self.n_unrollings == len(inputs)),\
                    'Number of unrollings does not coincide with '\
                    'number of inputs'

            super(SimpleRnn.RnnOutput, self)\
                .__init__(x0=x0, inputs=inputs, options=options, name=name)

            with tf.name_scope(self.scope):
                # network unrolling
                xt = self.x0
                x = list()
                y = list()
                self._unrolled = list()
                for t in range(n_unrollings):
                    x.append(xt)
                    yt, xt, net_t = self._next_step(xt, self.inputs[t], t)
                    y.append(yt)
                    self.unrolled.append(net_t)
                self._x = x
                self._y = y

    def evaluate(self, x0=None, inputs=None, n_unrollings=None, options=None,
                 name=None, **kargs):
        return type(self).RnnOutput(model=self, x0=x0, inputs=inputs,
                                    n_unrollings=n_unrollings,
                                    options=options, name=name)


class Rnn(tdl.core.TdlModel):
    @tdl.core.InputArgument
    def n_inputs(self, value):
        return value

    @tdl.core.InputArgument
    def n_outputs(self, value):
        return value

    def define_cell(self, n_inputs, n_outputs, n_states):
        pass

    @common.Submodel
    def cell(self, value):
        if isinstance(value, dict):
            return self.define_cell(**value)
        else:
            return value

    def __init__(self, n_inputs, n_outputs, n_states=None,
                 name='rnn', **kargs):
        '''
        @param n_inputs: number of exogenous inputs
        @param n_states: number of states
        @param n_outputs: number of observable outputs
        @param window_size: number of sequential inputs feed into the cell
        @param name: name used as the scope of the model
        '''
        if 'cell' not in kargs:
            kargs['cell'] = {'n_inputs': n_inputs, 'n_outputs': n_outputs,
                             'n_states': n_states}
        super(Rnn, self).__init__(
            n_inputs=n_inputs, n_outputs=n_outputs,
            name=name, **kargs)

    class RnnSetup(tdl.core.TdlModel):
        @tdl.core.InputModelInit(inference_input=True)
        def inputs(self, batch_size=None, AutoType=None):
            ''' exogenous inputs '''
            return self._setup_inputs(batch_size=batch_size,
                                      AutoType=AutoType)

        @tdl.core.InputModelInit(inference_input=True)
        def x0(self, batch_size=None, AutoType=None):
            ''' initial state '''
            value = self._define_x0(batch_size=batch_size,
                                    AutoType=AutoType)
            return value

        @tdl.core.InputArgument
        def n_unrollings(self, value):
            if value is None:
                assert tdl.core.is_property_set(self, 'inputs')
                value = len(self.inputs)
            return value

        @tdl.core.Submodel
        def _predictions(self, _):
            tdl.core.assert_initialized(self, '_predictions', ['n_unrollings'])
            x = list()
            y = list()
            unrolled = list()
            xt = self.x0
            for t in range(self.n_unrollings):
                yt, xt, net_t = self._next_step(xt, self.inputs[t], t)
                x.append(xt)
                y.append(yt)
                unrolled.append(net_t)
            return tdl.core.SimpleNamespace(outputs=y, states=x,
                                            unrolled=unrolled)

        @property
        def states(self):
            ''' State of the network '''
            return self._predictions.states

        @property
        def outputs(self):
            return self._predictions.outputs

        @property
        def unrolled(self):
            ''' list of unrolled networks '''
            return self._predictions.unrolled

        @property
        def loss(self):
            return self._loss

        @loss.setter
        def loss(self, value):
            if hasattr(self, '_loss'):
                warnings.warn('Model {} already has a loss defined'
                              ''.format(self))
            self._loss = value

        @property
        def n_inputs(self):
            ''' number of exogenous inputs '''
            return self.model.n_inputs

        @property
        def n_outputs(self):
            ''' number of outputs from the model '''
            return self.model.n_outputs

        @tdl.core.LazzyProperty
        def reset_inputs(self):
            return tf.variables_initializer(
                var_list=tdl.get_trainable(self.inputs))

        def _setup_inputs(self, batch_size=None, AutoType=None):
            ''' Setup either tf variables or placeholders for the external
            (control)  inputs '''
            tdl.core.assert_initialized(self, 'inputs', ['n_unrollings'])
            if AutoType is None:
                AutoType = tdl.core.autoinit.AutoPlaceholder()
            inputs = list()
            for k in range(self.n_unrollings):
                inputs_k = AutoType(
                    shape=(batch_size, self.n_inputs),
                    name='input_{}'.format(k))
                inputs.append(inputs_k)
            return inputs

        def _define_x0(self, batch_size, AutoType):
            ''' Define the initial state '''
            x0 = None
            raise NotImplementedError('x0 is not defined, please specify the '
                                      '_define_x0 method')
            return x0

        def _next_step(self, xt, ut, t):
            ''' Define the operation for computing the next step
            @param xk: current state, in case of narx, previous window_size
                       concatenated observations
            @param uk: current exogenous input
            @param k: current time step
            '''
            yt = None
            xt = None
            net = None
            raise NotImplementedError('No setup defined, please specify the '
                                      '_next_step method')
            return yt, xt, net

        def _define_regularizer(self, loss):
            return loss

        def _define_step_loss(self, yt, xt, ut, t, T):
            ''' Loss corresponding to step time t '''
            raise NotImplementedError('No step loss defined, either use '
                                      'compute_loss=False, or specify the '
                                      '_define_step_loss method')

        def _define_loss(self, y, x, u):
            step_loss = list()
            loss = 0
            with tf.name_scope('loss'):
                for t in range(len(y)):
                    loss_t = self._define_step_loss(
                        y[t], x[t], u[t], t, len(y))
                    step_loss.append(loss_t)
                loss = tdlf.AddNLosses(step_loss)
                fit_loss = tdlf.ScaledLoss(1.0/len(y), loss)
                if all([isinstance(l, tdlf.EmpiricalLoss) for l in loss]):
                    fit_loss = tdlf.EmpiricalLossWrapper(
                        loss=fit_loss,
                        labels=[l.labels for l in loss])
                loss = self._define_regularizer(fit_loss)
            return loss, fit_loss, step_loss

        def _init_options(self, options, default=None):
            default = {'regularizer/type': None,
                       'regularizer/coef': 0.0
                       }
            options = tdl.core.check_defaults(options, default)
            options = super(Rnn.RnnSetup, self)._init_options(options)
            return options

        @tdl.core.InputModel
        def model(self, value):
            return value

    def evaluate(self, x0=None, inputs=None, n_unrollings=None, options=None,
                 name=None, **kargs):
        if x0 is not None:
            kargs['x0'] = x0
        if inputs is not None:
            kargs['inputs'] = inputs
        if n_unrollings is not None:
            kargs['n_unrollings'] = n_unrollings
        return type(self).ModelOutput(model=self, options=options,
                                      name=name, **kargs)


class LstmState(tdl.core.TdlModel):
    @tdl.core.InferenceInput
    def x(self, value):
        return value

    @tdl.core.InferenceInput
    def h(self, value):
        return value

    def __init__(self, h, x):
        super(LstmState, self).__init__(h=h, x=x)


class LstmCellOptimized(tdl.core.TdlModel):
    ''' Single lstm cell defined as in: "Generating Sequences with
        Recurrent Neural Networks", Alex Graves, 2014

    Attributes:
        n_inputs: number of inputs
        n_nodes: number of nodes
        afunction: activation function
        name: name used in all TensorFlow variables' names
    '''
    @tdl.core.SimpleParameter
    def parameters(self, _):
        return [self.w_x, self.w_h,
                self.w_ci, self.w_cf,
                self.w_co, self.b_i,
                self.b_f, self.b_c,
                self.b_o]

    @property
    def weights(self):
        return [self.w_x, self.w_h,
                self.w_ci, self.w_cf,
                self.w_co]

    @tdl.core.InputArgument
    def n_inputs(self, value):
        return value

    @tdl.core.InputArgument
    def n_units(self, value):
        return value

    @property
    def n_outputs(self):
        return self.n_units

    @tdl.core.InputArgument
    def afunction(self, value):
        ''' activation function for the cell '''
        return value

    def __init__(self, n_inputs, n_units, afunction=tf.tanh, name='LstmCell'):
        super(LstmCellOptimized, self).__init__(
            n_inputs=n_inputs, n_units=n_units, afunction=afunction,
            name=name)

        # Note: w_ci, w_cf, wc_o are a diagonal matrices, therefore we only
        # reserve space for the diagonal, and the product with the cell state
        # is done as a element-by-element product instead of a matrix product
        with tf.name_scope(self.scope):
            self.w_x = tf.Variable(tf.truncated_normal(
                [n_inputs, 4 * n_units], stddev=0.1), name='w_xi')
            self.w_h = tf.Variable(tf.truncated_normal(
                [n_units, 4 * n_units], stddev=0.1), name='w_hi')
            # Input gate
            self.w_ci = tf.Variable(tf.truncated_normal(
                [n_units], stddev=0.1), name='w_ci')
            self.b_i = tf.Variable(tf.zeros([1, n_units]), name='b_i')
            # Forget gate: input, previous state, and bias.
            self.w_cf = tf.Variable(tf.truncated_normal(
                [n_units], stddev=0.1), name='w_cf')
            self.b_f = tf.Variable(tf.zeros([1, n_units]), name='b_f')
            # Memory cell: input, state and bias.
            self.b_c = tf.Variable(tf.zeros([1, n_units]), name='b_c')
            # Output gate: input, previous output, and bias.
            self.w_co = tf.Variable(tf.truncated_normal(
                [n_units], 0.0, 0.1), name='w_co')
            self.b_o = tf.Variable(tf.zeros([1, n_units]), name='b_o')

    class LstmCellSetup(tdl.core.OutputModel):
        @tdl.core.InferenceInput
        def inputs(self, value):
            return value

        @tdl.core.InferenceInput
        def input_state(self, value):
            return value

        @tdl.core.OutputValue
        def value(self, _):
            return tf.convert_to_tensor(self.y)

        @property
        def afunction(self):
            return self.model.afunction

    @tdl.core.ModelMethod(['y', 'state'], ['inputs', 'input_state'],
                          LstmCellSetup)
    def evaluate(self, object, inputs, input_state):
        """Create a LSTM cell.
        inputs: inputs, on the paper (Alex Graves) the notation is x_t
        state_h: hidden output on the previous time step, on the paper is h_t-1
        state_x: state on the previous time step, in the paper is c_t-1
        """
        input_h = input_state.h
        input_x = input_state.x
        # x transformation
        aux_x = tf.matmul(inputs, self.w_x)
        ix, fx, cx, ox = tf.split(aux_x, 4, axis=1)

        # h transformation
        aux_h = tf.matmul(input_h, self.w_h)
        ih, fh, ch, oh = tf.split(aux_h, 4, axis=1)

        # calculate gate values and output
        # input gate
        i_t = tf.sigmoid(ix + ih + input_x * self.w_ci + self.b_i)
        # forget gate
        f_t = tf.sigmoid(fx + fh + input_x * self.w_cf + self.b_f)
        update = cx + ch + self.b_c  # state update
        state = f_t * input_x + i_t * self.afunction(update)
        o_t = tf.sigmoid(ox + oh + state * self.w_co + self.b_o)
        full_state = LstmState(h=o_t * self.afunction(state),
                               x=state)
        return full_state.h, full_state


class MultilayerLstmCell(tdl.core.TdlModel):
    @property
    def parameters(self):
        params = list()
        for h in self.hidden_layers:
            params[len(params):] = h.parameters
        if self.output_layer is not None:
            params[len(params):] = self.output_layer.parameters
        return params

    @property
    def weights(self):
        params = list()
        for layer in self.hidden_layers:
            params[len(params):] = layer.weights
        if self.output_layer is not None:
            if isinstance(self.output_layer.weights, list):
                params[len(params):] = self.output_layer.weights
            else:
                params.append(self.output_layer.weights)
        return params

    @tdl.core.Submodel
    def hidden_layers(self, _):
        hidden = list()
        n_units = [self.n_inputs] + self.n_hidden
        for l in range(len(n_units) - 1):
            h = LstmCellOptimized(n_inputs=n_units[l],
                                  n_units=n_units[l+1],
                                  name='layer_{}'.format(l))
            hidden.append(h)
        return hidden

    @tdl.core.Submodel
    def output_layer(self, value):
        if isinstance(value, dict):
            output_layer = AffineLayer(
                n_inputs=self.hidden_layers[-1].n_outputs,
                **value)
        else:
            output_layer = value
        return output_layer

    @tdl.core.InputArgument
    def n_inputs(self, value):
        return value

    @tdl.core.InputArgument
    def n_hidden(self, value):
        value = ([value] if isinstance(value, int)
                 else value)
        assert isinstance(value, list),\
            'n_hidden must be a list with the number of hidden units'\
            'in each hidden layer'
        return value

    @property
    def n_outputs(self):
        return (self.output_layer.n_outputs
                if self.output_layer is not None
                else self.n_hidden[-1])

    def __init__(self, n_inputs, n_hidden, n_outputs=None,
                 output_layer=None, name=None,
                 **kargs):
        if n_outputs is not None:
            assert output_layer is None,\
                'n_outputs and output_layer cannot be both specified'
            output_layer = {'n_units': n_outputs}
        super(MultilayerLstmCell, self).__init__(
            n_inputs=n_inputs, n_hidden=n_hidden, output_layer=output_layer,
            name=name
        )

    class MultilayerLstmCellSetup(tdl.core.OutputModel):
        @tdl.core.InferenceInput
        def input_h(self):
            return self._in_x

        @tdl.core.InferenceInput
        def input_x(self):
            return self._in_x

        @property
        def state(self):
            return [h.state for h in self.hidden]

        @tdl.core.LazzyProperty
        def hidden(self):
            out = tf.convert_to_tensor(self.inputs)
            hidden = list()
            for l, layer in enumerate(self.model.hidden_layers):
                h = layer.evaluate(inputs=out,
                                   input_state=self.input_state[l])
                out = tf.convert_to_tensor(h)
                hidden.append(h)
            return hidden

        @tdl.core.LazzyProperty
        def output(self):
            if self.model.output_layer is not None:
                h = tf.convert_to_tensor(self.hidden[-1])
                output = self.model.output_layer.evaluate(h)
            else:
                output = self.hidden[-1]
            return output

    @tdl.core.ModelMethod(['y', 'value'], ['inputs', 'input_state'],
                          MultilayerLstmCellSetup)
    def evaluate(self, object, inputs, input_state):
        return object.output, tf.convert_to_tensor(object.output)


class Lstm(Rnn):
    @property
    def parameters(self):
        return self.cell.parameters

    @property
    def weights(self):
        return self.cell.weights

    @tdl.core.InputArgument
    def n_hidden(self, value):
        return value

    def define_cell(self, n_inputs, n_outputs, n_hidden):
        cell = MultilayerLstmCell(n_inputs=n_inputs,
                                  n_hidden=n_hidden,
                                  n_outputs=n_outputs)
        return cell

    def __init__(self, n_inputs, n_outputs, n_hidden,
                 name=None, **kargs):
        '''
        @param n_inputs: number of exogenous inputs
        @param n_hidden: list with the number of hidden units
        @param n_outputs: number of observable outputs
        @param name: name used as the scope of the model
        '''
        if 'cell' not in kargs:
            kargs['cell'] = {'n_inputs': n_inputs, 'n_outputs': n_outputs,
                             'n_hidden': n_hidden}
        super(Lstm, self).__init__(
            n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden,
            name=name, **kargs)

    class LstmSetup(Rnn.RnnSetup):
        class LstmStateAndOutput(tdl.core.TdlModel):
            @tdl.core.InferenceInput
            def hidden(self, value):
                return value

            @tdl.core.InferenceInput
            def y(self, value):
                return value

            def __init__(self, hidden, y):
                super(Lstm.LstmSetup.LstmStateAndOutput, self).\
                    __init__(hidden=hidden, y=y)

        @property
        def labels(self):
            return self._labels

        def _define_x0(self, batch_size=None, AutoType=None):
            if AutoType is None:
                AutoType = tdl.core.autoinit.AutoPlaceholder()
            state = list()
            y = None
            with tf.name_scope('x0'):
                if self.options['output/incremental']:
                    y = AutoType(shape=(batch_size, self.n_outputs),
                                 name='y')
                for l, layer in enumerate(self.model.cell.hidden_layers):
                    x = AutoType(shape=(batch_size, layer.n_units),
                                 name='x_{}'.format(l))
                    h = AutoType(shape=(batch_size, layer.n_units),
                                 name='h_{}'.format(l))
                    state.append(LstmState(h=h, x=x))
            return Lstm.LstmSetup.LstmStateAndOutput(hidden=state, y=y)

        def _next_step(self, xt, ut, t):
            net = self.model.cell.evaluate(inputs=ut,
                                           input_state=xt.hidden)
            if self.options['output/incremental']:
                yt = net.y + xt.y
            else:
                yt = net.y

            xt = Lstm.LstmSetup.LstmStateAndOutput(hidden=net.state, y=yt)
            return yt, xt, net

        def _define_regularizer(self, loss):
            if self.options['regularizer/type'] == 'l2':
                with tf.name_scope('regularizer'):
                    reg = tdlf.L2Regularizer(
                        weights=self.model.cell.weights,
                        scale=self.options['regularizer/scale'])
            else:
                raise ValueError('regularizer/type {} not implemented'
                                 ''.format(self.options['regularizer/type']))
            return tdlf.EmpiricalWithRegularization(empirical=loss,
                                                    regularizer=reg)

        def _define_step_loss(self, yt, xt, ut, t, T):
            with tf.name_scope('loss_{}'.format(t)):
                loss_t = tdlf.L2Loss(y=yt)
                self.labels.append(loss_t.labels)
            return loss_t

        def _init_options(self, options):
            default = {
                'regularizer/type': 'l2',
                'regularizer/scale': 1.0/0.00001,
                'mlp/keep_prob': None,
                'output/incremental': False
            }
            options = tdl.core.check_defaults(options, default)
            options = super(Lstm.LstmSetup, self)._init_options(options)
            return options

    ModelOutput = LstmSetup


class Lstm2Lstm(common.ModelBase):
    ''' Uses an Lstm to convert a fixed length sequence into the
    initial state for an LSTM sequential model '''

    @property
    def lstm(self):
        return self._lstm

    @property
    def encoder(self):
        return self._encoder

    @property
    def parameters(self):
        params = list()
        params += self.encoder.parameters
        params += self.lstm.parameters
        return params

    @property
    def weights(self):
        params = list()
        params += self.mlp.weights
        params += self.lstm.weights
        return params

    def __init__(self, n_inputs, n_outputs, n_hidden,
                 afunction=tf.tanh, encoder_afunction=tf.tanh,
                 name='lstm2lstm'):
        self._name = name

        with tf.name_scope(self.scope):
            self._encoder = Lstm(n_inputs=n_outputs,
                                 n_outputs=None,
                                 n_hidden=n_hidden,
                                 afunction=encoder_afunction,
                                 name='encoder_lstm')
            self._lstm = Lstm(n_inputs=n_inputs,
                              n_outputs=n_outputs,
                              n_hidden=n_hidden,
                              afunction=afunction,
                              name='lstm')

    class Output(common.ModelEvaluation):
        @property
        def encoder(self):
            return self._encoder

        @property
        def lstm(self):
            return self._lstm

        @property
        def x0(self):
            ''' Inputs to the encoder '''
            return self._encoder_inputs

        @property
        def loss(self):
            return self._loss

        @property
        def fit_loss(self):
            return self.lstm.fit_loss

        @property
        def labels(self):
            return self.lstm.labels

        @property
        def inputs(self):
            return self.lstm.inputs

        @property
        def y(self):
            return self.lstm.y

        def _define_x0(self, encoder_inputs, encoder_n_unrollings, batch_size):
            if encoder_inputs is None:
                encoder_inputs = list()
                for i in range(encoder_n_unrollings):
                    input_i = tf.placeholder(tf.float32,
                                             shape=(batch_size,
                                                    self.model.lstm.n_outputs),
                                             name='x0_{}'.format(i))
                    encoder_inputs.append(input_i)
            self.placeholders.x0 = encoder_inputs
            if batch_size is None:
                batch_size = tf.shape(encoder_inputs[0])[0]
            encoder = self.model.encoder.setup(inputs=encoder_inputs,
                                               n_unrollings=encoder_n_unrollings,
                                               batch_size=batch_size,
                                               compute_loss=False,
                                               options={'x0/type': 'zeros'})

            lstm_x0 = Lstm.LstmSetup.LstmState(
                h=[layer.h for layer in encoder.unrolled[-1].hidden],
                x=[layer.x for layer in encoder.unrolled[-1].hidden],
                y=encoder_inputs[-1])

            return encoder_inputs, lstm_x0, encoder

        def _setup_lstm(self, x0, n_unrollings, batch_size, inputs,
                        compute_loss, options):
            lstm = self.model.lstm.setup(x0=x0,
                                         n_unrollings=n_unrollings,
                                         batch_size=batch_size,
                                         inputs=inputs,
                                         compute_loss=compute_loss,
                                         options=options)
            if hasattr(lstm.placeholders, 'inputs'):
                self.placeholders.inputs = lstm.placeholders.inputs
            return lstm

        def _define_loss(self, encoder, lstm):
            ''' Define the loss for fitting the model '''
            with tf.name_scope('loss'):
                with tf.name_scope('regularization'):
                    l2w = [tf.nn.l2_loss(w)
                           for w in encoder.model.weights]
                    mlp_reg = tf.add_n(l2w)

                loss = lstm.loss + self.options['regularizer/coef'] * mlp_reg
            return loss

        def _init_options(self, options):
            default = {
                'regularizer/type': 'l2',
                'regularizer/coef': 0.00001
            }
            options = common.ModelEvaluation\
                            ._init_options(self, options, default)
            return options

        def __init__(self, model, n_unrollings=1, encoder_n_unrollings=1,
                     batch_size=None, inputs=None, encoder_inputs=None,
                     compute_loss=True, options=None, name='lstm2lstm'):
            super(Lstm2Lstm.Output, self)\
                .__init__(model, options=options, name=name)

            with tf.name_scope(self.scope):
                self._encoder_inputs, self._lstm_x0, self._encoder = \
                    self._define_x0(encoder_inputs,
                                    encoder_n_unrollings, batch_size)
                self._lstm = self._setup_lstm(x0=self._lstm_x0,
                                              n_unrollings=n_unrollings,
                                              batch_size=batch_size,
                                              inputs=inputs,
                                              compute_loss=compute_loss,
                                              options=options)
                if compute_loss:
                    self._loss = self._define_loss(self.encoder, self.lstm)
    ModelOutput = Output


class Mlp2Lstm(tdl.core.TdlModel):
    ''' Uses an MLP to convert a fixed length sequence into the
    initial state for an LSTM sequential model '''

    @tdl.core.InputArgument
    def n_inputs(self, value):
        return value

    @tdl.core.InputArgument
    def n_outputs(self, value):
        return value

    @tdl.core.InputArgument
    def window_size(self, value):
        return value

    @tdl.core.SubmodelInit
    def lstm(self, n_hidden, **kargs):
        return Lstm(n_inputs=self.n_inputs,
                    n_outputs=self.n_outputs,
                    n_hidden=n_hidden,
                    **kargs)

    @tdl.core.SubmodelInit
    def mlp(self, n_hidden, **kargs):
        assert tdl.core.is_property_set(self, 'lstm'),\
            'lstm has not been initialized. Please initialize Lstm first'
        return tdlf.MlpNet(n_inputs=self.n_outputs * self.window_size,
                           n_outputs=2 * sum(self.lstm.n_hidden),
                           n_hidden=n_hidden,
                           **kargs)

    @property
    def parameters(self):
        params = list()
        params += self.mlp.parameters
        params += self.lstm.parameters
        return params

    @property
    def weights(self):
        params = list()
        params += self.mlp.weights
        params += self.lstm.weights
        return params

    def __init__(self, n_inputs, n_outputs, window_size=1,
                 name=None, **kargs):
        """ Lstm model with initial state computed using an MLP.
        Args:
            n_inputs (int): number of external inputs.
            n_outputs (int): number of target outputs.
            window_size (int): window size for the initial state. Defaults to 1.
            name (str): name of the model. Defaults to None.
            **kargs (): other parameters used to initialize the model.
        Returns:
            Mlp2Lstm: TdlModel.
        """
        super(Mlp2Lstm, self).__init__(
            n_inputs=n_inputs, n_outputs=n_outputs,
            window_size=window_size)

    class Mlp2LstmOutput(tdl.core.OutputModel):
        @tdl.core.InputArgument
        def batch_size(self, value):
            if value is None:
                if (tdl.core.is_property_set(self, 'inputs') and
                        tdl.core.is_property_set(self, 'x0')):
                    input_batch = tf.convert_to_tensor(self.inputs[0])\
                                    .shape[0].value
                    x0_batch = tf.convert_to_tensor(self.x0[0])\
                                 .shape[0].value
                    if (input_batch == x0_batch):
                        return input_batch
                    else:
                        raise ValueError(
                            'batch_size for x0 and inputs are different. '
                            'Refer to them sepearatelly')
                else:
                    raise tdl.core.exceptions.InitPreconditionsFailed(
                        object=self, property='batch_size',
                        reqs=['inputs', 'x0'])
            return value

        @property
        def n_unrollings(self):
            return len(self.inputs)

        @property
        def window_size(self):
            return len(self.x0)

        @tdl.core.SubmodelInit
        def inputs(self, n_unrollings, Type='placeholder', batch_size=None):
            if batch_size is not None:
                self.batch_size = batch_size
            elif tdl.core.is_property_set(self, 'batch_size'):
                batch_size = self.batch_size
            if isinstance(Type, str):
                if Type == 'placeholder':
                    Type = lambda **kargs: tf.placeholder(**kargs)
                elif Type == 'zeros':
                    Type = lambda **kargs: tf.zeros(**kargs)
                else:
                    raise ValueError('unrecognized type {}'.format(type))
            value = [Type(dtype=tdl.core.global_options.float.tftype,
                          shape=[batch_size, self.model.n_inputs])
                     for i in range(n_unrollings)]
            return value

        @tdl.core.SubmodelInit
        def x0(self, Type='placeholder', batch_size=None):
            if batch_size is not None:
                self.batch_size = batch_size
            elif tdl.core.is_property_set(self, 'batch_size'):
                batch_size = self.batch_size
            if isinstance(Type, str):
                if Type == 'placeholder':
                    Type = lambda **kargs: tf.placeholder(**kargs)
                elif Type == 'zeros':
                    Type = lambda **kargs: tf.zeros(**kargs)
                else:
                    raise ValueError('unrecognized type {}'.format(type))
            value = [Type(dtype=tdl.core.global_options.float.tftype,
                          shape=[batch_size, self.model.n_outputs])
                     for i in range(self.model.window_size)]
            return value

        @tdl.core.Submodel
        def lstm_x0(self, _):
            tdl.core.assert_initialized(self, 'lstm_x0', ['x0'])
            mlp_inputs = tf.concat(self.x0, axis=1)
            mlp = self.model.mlp.evaluate(inputs=mlp_inputs)
            lstm_x0_h, lstm_x0_x = tf.split(tf.convert_to_tensor(mlp), 2,
                                            axis=1)
            h = tf.split(lstm_x0_h, len(self.model.lstm.n_hidden), axis=1)
            x = tf.split(lstm_x0_x, len(self.model.lstm.n_hidden), axis=1)
            state = [LstmState(h=hi, x=xi) for hi, xi in zip(h, x)]
            return Lstm.LstmSetup.LstmStateAndOutput(hidden=state,
                                                     y=self.x0[-1])

        @tdl.core.Submodel
        def lstm(self, _):
            tdl.core.assert_initialized(self, 'lstm', ['x0', 'inputs'])
            lstm = self.model.lstm.evaluate(
                x0=self.lstm_x0, inputs=self.inputs)
            return lstm

    def evaluate(self, x0=None, inputs=None, **kargs):
        if x0 is not None:
            kargs['x0'] = x0
        if inputs is not None:
            kargs['inputs'] = inputs
        return Mlp2Lstm.Mlp2LstmOutput(model=self, **kargs)


class Narx(Rnn):
    @property
    def window_size(self):
        return self._window_size

    def define_cell(self, n_inputs, n_outputs, window_size):
        pass

    @common.Submodel
    def cell(self, value):
        if value is None:
            return self.define_cell(n_inputs=self.n_inputs,
                                    n_outputs=self.n_outputs,
                                    window_size=self.window_size)
        else:
            return value

    def __init__(self, n_inputs, n_outputs, window_size=1,
                 name='narx', **kargs):
        '''
        @param n_inputs: number of exogenous inputs
        @param n_outputs: number of observable outputs
        @param window_size: number of sequential inputs feed into the cell
        @param name: name used as the scope of the model
        '''
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._window_size = window_size
        common.TdlModel.__init__(self, name=name, **kargs)

    class NarxSetup(Rnn.RnnSetup):
        @property
        def window_size(self):
            return self.model.window_size


class MlpNarx(Narx):
    ''' Narx that uses an Mlp as cell '''
    CellModel = tdlf.MlpNet

    @property
    def n_hidden(self):
        ''' Number of hidden layers '''
        return self._n_hidden

    @property
    def afunction(self):
        ''' activation function for the MLP '''
        return self._afunction

    def define_cell(self, n_inputs, n_outputs, window_size):
        cell = self.CellModel(n_inputs=n_inputs + n_outputs * window_size,
                              n_outputs=n_outputs,
                              n_hidden=self.n_hidden,
                              afunction=self.afunction,
                              name='cell')
        return cell

    def __init__(self, n_inputs, n_outputs, window_size,
                 n_hidden, afunction=tf.nn.relu,
                 name='mlp_narx', **kargs):
        self._n_hidden = n_hidden
        self._afunction = afunction

        super(MlpNarx, self).__init__(
            n_inputs=n_inputs, n_outputs=n_outputs, window_size=window_size,
            name=name, **kargs)

    class Output(Narx.NarxSetup):
        @property
        def labels(self):
            return self._labels

        def _define_x0(self, batch_size):
            x0 = list()
            for i in range(self.window_size):
                x0.append(tf.placeholder(tf.float32,
                                         shape=(self.options['x0/batch_size'],
                                                self.n_outputs),
                                         name='x0_{}'.format(i)))
            self.placeholders.x0 = x0
            return tuple(x0)

        def _xt_transfer_func(self, xt, t):
            ''' Defines gx for x[t+1] = f(gx(x[t]), gu(u[t])) '''
            return xt

        def _ut_transfer_func(self, ut, t):
            ''' Defines gu for x[t+1] = f(gx(x[t]), gu(u[t])) '''
            return ut

        def _next_step(self, xt, ut, t):
            ''' Defines f for x[t+1] = f(gx(x[t]), gu(u[t])) '''
            with tf.name_scope('cell_{}'.format(t)):
                xt = self._xt_transfer_func(xt, t)
                ut = self._ut_transfer_func(ut, t)
                net = self.model.cell.evaluate(
                    tf.concat(list(xt) + [ut], axis=1))
                yt = net.y + xt[-1]
                xt = tuple(list(xt[1:]) + [yt])
            return yt, xt, net

        def _define_regularizer(self, loss):
            ''' return loss + alpha*regularizer '''
            if self.options['regularizer/type'] == 'l2':
                with tf.name_scope('regularizer'):
                    reg = tdlf.L2Regularizer(
                        weights=self.model.cell.weights,
                        scale=self.options['regularizer/scale'])
            else:
                raise ValueError('regularizer/type {} not implemented'
                                 ''.format(self.options['regularizer/type']))
            return tdlf.EmpiricalWithRegularization(empirical=loss,
                                                    regularizer=reg)

        def _define_step_loss(self, yt, xt, ut, t, T):
            ''' fit loss at timestep t '''
            with tf.name_scope('loss_{}'.format(t)):
                loss_t = tdlf.L2Loss(y=yt)
                self.labels.append(loss_t.labels)
            return loss_t

        def _init_options(self, options):
            default = {
                'regularizer/type': 'l2',
                'regularizer/scale': 1/0.00001,
                'mlp/keep_prob': None,
                'x0/batch_size': self.batch_size
            }
            options = common.ModelEvaluation\
                            ._init_options(self, options, default)
            options = super(MlpNarx.Output, self)._init_options(options)
            return options

        def __init__(self, model, x0=None, n_unrollings=1, batch_size=None,
                     inputs=None, compute_loss=True, options=None, name=None):

            if compute_loss:
                self._labels = list()

            super(MlpNarx.Output, self).__init__(
                model=model, x0=x0, n_unrollings=n_unrollings,
                batch_size=batch_size, inputs=inputs,
                compute_loss=compute_loss, options=options, name=name)
    ModelOutput = Output

    def setup(self, *args, **kargs):
        warnings.warn('setup is deprecated, will be removed in the future')
        assert len(args) == 0,\
            'arguments for setup must be explicitly specified'
        return self.ModelOutput(self, **kargs)
