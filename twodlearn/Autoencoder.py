"""Definition of autoencoder layers and a stacked autoencoder network
"""

import tensorflow as tf
from . import core
from . import feedforward as tdlf


class TransposedAffine(tdlf.AffineLayer):
    def __init__(self, reference_layer, name=None):
        self.reference_layer = reference_layer
        if name is None:
            name = reference_layer.name + 'transposed'

        self._name = name
        self._n_inputs = reference_layer.n_units
        self._n_units = reference_layer.n_inputs
        with tf.name_scope(self.scope):
            self.weights = tf.transpose(reference_layer.weights)
            self.bias = tf.Variable(tf.zeros([self.n_units]),
                                    name='b')


class TransposedFullyConnected(tdlf.DenseLayer):
    def __init__(self, reference_layer, afunction=None, name=None):
        self.reference_layer = reference_layer
        if name is None:
            name = reference_layer.name + 'transposed'
        if afunction is None:
            afunction = reference_layer.afunction

        self._name = name
        self._n_inputs = reference_layer.n_units
        self._n_units = reference_layer.n_inputs
        self.afunction = afunction
        with tf.name_scope(self.scope):
            self.weights = tf.transpose(reference_layer.weights)
            self.bias = tf.Variable(tf.zeros([self.n_units]),
                                    name='b')


class TransposedMlpNet(tdlf.MlpNet):
    def define_fullyconnected_layers(self):
        ''' Defines the model for the fully connected layers '''
        full_layers = list()
        full_layers.append(TransposedFullyConnected(self.reference_model.out_layer,
                                                    afunction=self.afunction[-1],
                                                    name='full_0'))
        for l, layer in enumerate(reversed(
                self.reference_model.full_layers[1:])):
            full_layers.append(
                TransposedFullyConnected(layer, name='full_{}'.format(l)))
        return full_layers

    def define_output_layer(self, output_function):
        ''' Defines the model for the final layer '''
        if output_function is None:
            out_layer = TransposedAffine(self.reference_model.full_layers[0],
                                         name='output_layer')
        else:
            out_layer = TransposedFullyConnected(self.reference_model.full_layers[0],
                                                 afunction=output_function,
                                                 name='output_layer')
        return out_layer

    def __init__(self, encoder_net, output_function=None, name=None):
        ''' Creates a feedforward network with weights tied to encoder_net
        @param encoder_net: Encoder network to be transposed
        '''
        self.reference_model = encoder_net
        n_inputs = encoder_net.n_outputs
        n_outputs = encoder_net.n_inputs
        n_hidden = encoder_net.n_hidden[::-1]
        afunction = encoder_net.afunction
        if name is None:
            name = encoder_net.name + '_transposed'
        super(TransposedMlpNet, self).__init__(
            n_inputs, n_outputs, n_hidden, afunction,
            output_function, name=name)


class AutoencoderNet(core.TdlModel):
    @property
    def parameters(self):
        params = self.encoder.parameters
        if not self.tied_weights:
            params += self.decoder.parameters
        return params

    @property
    def tied_weights(self):
        return self._tied_weights

    def define_encoder(self, n_inputs, enc_hidden, n_outputs, hidden_afunction,
                       output_function):
        encoder = tdlf.MlpNet(n_inputs=n_inputs,
                              n_outputs=n_outputs,
                              n_hidden=enc_hidden,
                              afunction=hidden_afunction,
                              output_function=output_function,
                              name='encoder')
        return encoder

    def define_decoder(self, n_inputs, dec_hidden, n_outputs, hidden_afunction,
                       output_function):
        if self.tied_weights:
            decoder = TransposedMlpNet(self.encoder,
                                       output_function=output_function,
                                       name='decoder')
        else:
            decoder = tdlf.MlpNet(n_inputs=n_inputs,
                                  n_outputs=n_outputs,
                                  n_hidden=dec_hidden,
                                  afunction=hidden_afunction,
                                  output_function=output_function,
                                  name='decoder')
        return decoder

    def __init__(self, n_inputs, enc_hidden, enc_out,
                 hidden_afunction=tf.nn.relu,
                 dec_hidden=None, output_function=None,
                 enc_output_function=None,
                 tied_weights=False,
                 name='autoencoder'):
        '''All variables corresponding to the weights of the network are defined
        '''
        self._name = name
        self._n_inputs = n_inputs
        self._tied_weights = tied_weights
        if dec_hidden is None:
            dec_hidden = enc_hidden[::-1]

        with tf.name_scope(self.scope):
            self.encoder = self.define_encoder(n_inputs,
                                               enc_hidden,
                                               enc_out,
                                               hidden_afunction,
                                               enc_output_function)
            self.decoder = self.define_decoder(enc_out,
                                               dec_hidden,
                                               n_inputs,
                                               hidden_afunction,
                                               output_function)

    class AutoencoderNetSetup(core.TdlModel):
        @property
        def n_inputs(self):
            return self.model.n_inputs

        @property
        def tied_weights(self):
            return self.model.tied_weights

        @property
        def contractive_loss(self):
            ''' mean squared frobenius norm of the jacobian'''
            if self.opt['regularization/contractive'] is True:
                return self._contractive_loss
            else:
                return 0.0

        @property
        def jacobian_f(self):
            ''' mean frobenius norm of the jacobian'''
            if self.opt['regularization/contractive'] is True:
                return self._jacobian_f
            else:
                return None

        @property
        def wd_loss(self):
            ''' weight-decay, ussually a frobenius norm of the weights '''
            return self._wd_loss

        @property
        def reconstruction_loss(self):
            ''' loss that penalize for the reconstructions of the original
            inputs '''
            return self._reconstruction_loss

        def add_noise(self, inputs, opt):
            ''' adds noise to the input

            Args:
                inputs: intpus for which noise will be added
                opt: dictionary with the options for the autoencoder,
                    the options regarding noise are:
                    'noise/type': 'bernoulli', 'gaussian'
                    'noise/level': float (0.0-1.0 typically) specifing how much
                    noise will be added
            @return noisy_inputs
            '''
            if opt['noise/type'] is None:
                return inputs
            elif opt['noise/type'] == 'bernoulli':
                return tf.nn.dropout(inputs, 1 - opt['noise/level'])
            elif opt['noise/type'] == 'gaussian':
                input_shape = tf.shape(inputs)
                dist = tf.distributions.Normal(loc=[0.0] * self.n_inputs,
                                               scale=[1.0] * self.n_inputs)
                return inputs + opt['noise/level'] * dist.sample([input_shape[0]])
                # return inputs + opt['noise/level'] * tf.random_normal(shape=input_shape)

        def define_encoder_forward(self, inputs):
            encoder = self.model.encoder.setup(batch_size=self.batch_size,
                                               keep_prob=self.keep_prob,
                                               l2_reg_coef=None,
                                               loss_type=None,
                                               inputs=inputs,
                                               name='encoder')
            return encoder

        def define_decoder_forward(self, inputs):
            decoder = self.model.decoder.setup(batch_size=self.batch_size,
                                               keep_prob=self.keep_prob,
                                               l2_reg_coef=None,
                                               loss_type=None,
                                               inputs=inputs,
                                               name='decoder')
            return decoder

        def define_reconstruction_loss(self, inputs, decoder, opt):
            if opt['reconstruction_loss_type'] == 'sigmoid':
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,
                                                                              logits=decoder.output.z))
            elif opt['reconstruction_loss_type'] == 'l2':
                loss = tf.reduce_mean(
                    tf.pow(inputs - decoder.y, 2))
            else:
                raise ValueError(
                    'reconstruction_loss_type must be sigmoid or l2')
            return loss

        def setup_contractive_regularizer(self):
            reg = 0.0
            mean_jacobian = 0.0
            for layer in self.encoder.hidden + [self.encoder.output]:
                if type(layer) is tdlf.AffineLayerOutput:
                    reg += tf.nn.l2_loss(layer.weights)
                    mean_jacobian += tf.sqrt(tf.nn.l2_loss(layer.weights))

                elif layer.afunction is tf.nn.relu:
                    z = tf.stop_gradient(layer.z)
                    df_dz = tf.cast(z > 0.0, tf.float32)
                    frob = tf.reduce_sum((df_dz**2.0) * tf.reduce_sum(layer.weights**2.0, 0),
                                         axis=1)
                    reg += tf.reduce_mean(frob,
                                          name='contractive_for_relu')
                    mean_jacobian += tf.reduce_mean(tf.sqrt(frob))
                elif layer.afunction is tf.nn.sigmoid:
                    if self.opt['regularization/contractive/stop_gradient']:
                        x = tf.stop_gradient(layer.inputs)
                        h = tf.nn.sigmoid(
                            tf.matmul(x, layer.weights) + layer.bias)
                    else:
                        h = layer.y
                    df_dz = h * (1 - h)
                    frob = tf.reduce_sum((df_dz**2.0) * tf.reduce_sum(layer.weights**2.0, 0),
                                         axis=1)
                    frob = frob
                    reg += tf.reduce_mean(frob,
                                          name='contractive_for_sigmoid')
                    mean_jacobian += tf.reduce_mean(tf.sqrt(frob))
                    # self.test1 = tf.reduce_mean(frob,
                    #                             name='contractive_for_sigmoid')
                    # ---- computing the jacobian explicitely ---- #
                    # h = tf.stop_gradient(layer.y)
                    # df_dz = h * (1 - h)
                    # J = tf.reshape(df_dz, (-1, 1, layer.n_units)) * \
                    #     tf.reshape(layer.weights,
                    #                (1, layer.n_inputs, layer.n_units))
                    # frob = tf.reduce_mean(tf.reduce_sum(
                    #     J**2.0, axis=(1, 2)), axis=0)
                    # reg += tf.reduce_mean(frob,
                    #                       name='contractive_for_sigmoid')

                    # self.test2 = tf.reduce_mean(frob,
                    #                             name='contractive_for_sigmoid')

                elif layer.afunction is tf.nn.softplus:
                    # print('adding contractive regularizer for softplus')
                    if self.opt['regularization/contractive/stop_gradient']:
                        x = tf.stop_gradient(layer.inputs)
                        z = tf.matmul(x, layer.weights) + layer.bias
                    else:
                        z = layer.z  # tf.stop_gradient(layer.z)
                    df_dz = tf.sigmoid(z)
                    frob = tf.reduce_sum((df_dz**2.0) * tf.reduce_sum(layer.weights**2.0, 0),
                                         axis=1)
                    #frob = tf.sqrt(frob)
                    reg += tf.reduce_mean(frob,
                                          name='contractive_for_softplus')
                    mean_jacobian += tf.reduce_mean(tf.sqrt(frob))

            self._contractive_loss = reg / (len(self.encoder.hidden) + 1)
            self._jacobian_f = mean_jacobian / (len(self.encoder.hidden) + 1)
            return self._contractive_loss

        def define_regularizer(self):
            self._wd_loss = None
            self._contractive_loss = None
            self._jacobian_f = None

            weights_list = self.weights
            regularizers = list()
            # Weight decay
            if self.opt['regularization/weights/type'] == 'l2':
                w_reg = [tf.sqrt(tf.nn.l2_loss(w)) for w in weights_list]
                w_reg = tf.add_n(w_reg)
                regularizers.append(
                    self.opt['regularization/weights/coef'] * w_reg)
                self._wd_loss = w_reg
            elif self.opt['regularization/weights/type'] == 'sl2':
                w_reg = [tf.nn.l2_loss(w) for w in weights_list]
                w_reg = tf.add_n(w_reg)
                regularizers.append(
                    self.opt['regularization/weights/coef'] * w_reg)
                self._wd_loss = w_reg
            else:
                raise ValueError("weight regularization invalid")
            # contractive
            if self.opt['regularization/contractive'] is True:
                contractive = self.setup_contractive_regularizer()
                regularizers.append(
                    self.opt['regularization/contractive/coef'] * contractive)
            # Sum of all regularizers
            if not regularizers:
                reg_loss = None
            else:
                reg_loss = tf.add_n(regularizers)
            return reg_loss

        def define_supervised_loss(self):
            pass

        def define_opt(self, opt):
            if 'reconstruction_loss_type' not in opt:
                opt['reconstruction_loss_type'] = 'sigmoid'
            if 'regularization/weights/type' not in opt:
                opt['regularization/weights/type'] = 'l2'
            if 'regularization/weights/coef' not in opt:
                opt['regularization/weights/coef'] = 1e-5
            if 'regularization/contractive' not in opt:
                opt['regularization/contractive'] = False
            if 'regularization/contractive/coef' not in opt:
                opt['regularization/contractive/coef'] = 1e-5
            if 'regularization/contractive/stop_gradient' not in opt:
                opt['regularization/contractive/stop_gradient'] = True
            if 'noise/type' not in opt:
                opt['noise/type'] = None
            if 'noise/level' not in opt:
                opt['noise/level'] = 0.0
            return opt

        @property
        def model(self):
            ''' model that holds the parameters '''
            return self._model

        @property
        def name(self):
            ''' name used for the construction of the comptuation graph '''
            return self._name

        @property
        def opt(self):
            ''' options used to build the graph '''
            return self._opt

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def keep_prob(self):
            return self._keep_prob

        @property
        def encoder(self):
            return self._encoder

        @property
        def decoder(self):
            return self._decoder

        @property
        def loss(self):
            ''' Reconstruction loss + regularizers '''
            return self._loss

        @property
        def inputs(self):
            ''' inputs to the encoder '''
            return self._inputs

        @property
        def y(self):
            ''' outputs from the decoder '''
            return self.decoder.y

        @property
        def weights(self):
            weights = list()
            weights += self.encoder.weights
            if not self.tied_weights:
                weights += self.decoder.weights
            return weights

        def __init__(self, model, batch_size=None, keep_prob=None,
                     inputs=None, opt=None, name='train'):
            self._name = name
            self._model = model
            if opt is None:
                self._opt = self.define_opt(dict())
            else:
                self._opt = self.define_opt(opt)
            self._batch_size = batch_size
            self._keep_prob = keep_prob

            with tf.name_scope(self.scope):
                if inputs is None:
                    inputs = tf.placeholder(tf.float32,
                                            shape=(self.batch_size,
                                                   self.n_inputs),
                                            name='inputs')
                self._inputs = inputs
                noisy_inputs = self.add_noise(self.inputs, self.opt)
                self._encoder = self.define_encoder_forward(noisy_inputs)
                self._decoder = self.define_decoder_forward(self.encoder.y)

                self._reconstruction_loss = self.define_reconstruction_loss(self.inputs,
                                                                            self.decoder,
                                                                            self.opt)
                loss = self.reconstruction_loss
                regularization_loss = self.define_regularizer()
                if regularization_loss is not None:
                    loss += regularization_loss

                self._loss = loss

    def setup(self, batch_size=None, keep_prob=None, inputs=None,
              opt=None, name=None):
        if name is None:
            name = self.name
        return self.AutoencoderNetSetup(self, batch_size=batch_size,
                                        keep_prob=keep_prob,
                                        inputs=inputs,
                                        opt=opt,
                                        name=name)


class AutoencoderClassifierNet(AutoencoderNet):
    ''' Autoencoder with a linear classifier in the encoder output '''

    @property
    def parameters(self):
        params = self.encoder.parameters
        if not self.tied_weights:
            params += self.decoder.parameters
        params += self.classifier.parameters
        return params

    def define_classifier(self, n_classes):
        classifier = tdlf.LinearClassifier(
            self.encoder.n_outputs, self.n_classes)
        return classifier

    def __init__(self, n_inputs, enc_hidden, enc_out, n_classes,
                 hidden_afunction=tf.nn.relu, dec_hidden=None,
                 output_function=None, enc_output_function=None,
                 tied_weights=False,
                 name='autoencoder'):
        super(AutoencoderClassifierNet, self).__init__(
            n_inputs=n_inputs, enc_hidden=enc_hidden,
            enc_out=enc_out, hidden_afunction=hidden_afunction,
            dec_hidden=dec_hidden, output_function=output_function,
            enc_output_function=enc_output_function,
            tied_weights=tied_weights, name=name)
        self.n_classes = n_classes
        self.classifier = self.define_classifier(self.n_classes)

    class AutoencoderClassifierNetSetup(AutoencoderNet.AutoencoderNetSetup):
        def setup_classifier(self):
            with tf.name_scope(self.scope):
                classifier = self.model.classifier.setup(
                    batch_size=self.batch_size, inputs=self.encoder.y,
                    name='classifier')
            return classifier

        def __init__(self, model, batch_size=None, keep_prob=None,
                     inputs=None, opt=None, name='train'):
            super(AutoencoderClassifierNet.AutoencoderClassifierNetSetup,
                  self).__init__(model=model, batch_size=batch_size,
                                 keep_prob=keep_prob, inputs=inputs, opt=opt, name=name)
            self.classifier = self.setup_classifier()

    def setup(self, batch_size=None, keep_prob=None,
              inputs=None, opt=None, name=None):
        if name is None:
            name = self.name
        return self.AutoencoderClassifierNetSetup(self, batch_size=batch_size,
                                                  keep_prob=keep_prob,
                                                  inputs=inputs,
                                                  opt=opt,
                                                  name=name)


class AutoencoderNetConf(object):
    '''This is a wrapper to any network configuration, it contains the references to
    the placeholders for inputs and labels, and the reference of the computation graph for
    the network

    inputs: placeholder for the inputs
    labels: placeholder for the labels
    y: output of the comptuation graph (logits)
    loss: loss for the network

    xp_list: list with the "reconstructed" outputs using the transpose of the autoencoder layer
    h_list: list with the hidden layer outputs
    xp_losses: list with the loses for each one of the layers reconstructions
    '''

    def __init__(self, inputs, labels, y, loss, xp_list, h_list, xp_losses):
        self.inputs = inputs
        self.labels = labels
        self.y = y
        self.loss = loss

        self.xp_list = xp_list
        self.h_list = h_list
        self.xp_losses = xp_losses


class StackedAutoencoderNet(object):
    def __init__(self, n_inputs, n_outputs, n_hidden, afunction=None, name=''):
        '''All variables corresponding to the weights of the network are defined
        n_inputs: number of inputs
        n_outputs: number of outputs
        n_hidden: list with the number of hidden units in each layer
        full_layers: list of fully connected layers
        afunction: function, or list of functions specifying the activation function being used. if
                   not specified, the default is relu
        out_layer: output layer, for the moment, linear layer


        '''
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # Check activation function being used
        if afunction is not None:
            if isinstance(afunction, list):
                self.afunction = afunction
            else:
                self.afunction = [afunction for i in range(len(n_hidden))]
        else:
            self.afunction = [tf.sigmoid for i in range(len(n_hidden))]

        # Create the fully connected layers:
        self.full_layers = list()
        self.full_layers.append(
            FullyconnectedLayer(n_inputs, n_hidden[0],
                                afunction=self.afunction[0],
                                name='0_full_' + name)
        )
        for l in range(1, len(n_hidden)):
            self.full_layers.append(
                AutoencoderLayer(n_hidden[l - 1], n_hidden[l],
                                 afunction=self.afunction[l],
                                 name=str(l) + '_full_' + name)
            )

        # Create the final layer:
        self.out_layer = LinearLayer(
            n_hidden[-1], n_outputs, name='_lin_' + name)

        # 4. Define the saver for the weights of the network
        saver_dict = dict()
        for l in range(len(self.full_layers)):
            saver_dict.update(self.full_layers[l].saver_dict)

        saver_dict.update(self.out_layer.saver_dict)

        self.saver = tf.train.Saver(saver_dict)

    def compute_pred_loss(self, x, xp):
        return tf.reduce_mean(tf.nn.l2_loss(x - xp))

    def compute_output_loss(self, y, labels):
        # for cross-entropy loss
        # if self.n_outputs==1:
        #    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, labels)) + l2_reg
        # else:
        #    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels)) + l2_reg

        # for l2 loss:
        return tf.reduce_mean(tf.nn.l2_loss(y - labels))

    def add_noise(self, input_tensor, l_idx):
        return input_tensor

    def setup(self, batch_size, drop_prob=None, l2_reg_coef=None, inputs=None):
        ''' Defines the computation graph of the neural network for a specific batch size

        drop_prob: placeholder used for specify the probability for dropout. If this coefficient is set, then
                   dropout regularization is added between all fully connected layers(TODO: allow to choose which layers)
        l2_reg_coef: coeficient for l2 regularization
        loss_type: type of the loss being used for training the network, the options are:
                - 'cross_entropy': for classification tasks
                - 'l2': for regression tasks
        '''
        if inputs is None:
            inputs = tf.placeholder(tf.float32,
                                    shape=(batch_size, self.n_inputs))

        labels = tf.placeholder(tf.float32, shape=(batch_size, self.n_outputs))

        # 1. forward computation stage
        h_list = list()
        xp_list = list()

        h_list.append(self.full_layers[0].evaluate(self.add_noise(inputs, 0)))
        xp_list.append(self.full_layers[0].evaluate_transpose(h_list[-1]))

        for l_idx in range(1, len(self.full_layers)):

            if drop_prob is not None:
                h_list.append(
                    self.full_layers[l_idx].evaluate(
                        tf.nn.dropout(
                            self.add_noise(h_list[-1], l_idx),
                            drop_prob))
                )
            else:
                h_list.append(self.full_layers[l_idx].evaluate(
                    self.add_noise(h_list[-1], l_idx)))

            xp_list.append(
                self.full_layers[l_idx].evaluate_transpose(h_list[-1]))

        # 2. linear stage
        if drop_prob is not None:
            y = self.out_layer.evaluate(tf.nn.dropout(h_list[-1], drop_prob))
        else:
            y = self.out_layer.evaluate(h_list[-1])

        # 3. loss
        # l2 regularizer
        l2_reg = 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights)
            l2_reg += tf.nn.l2_loss(self.out_layer.weights)
            l2_reg = l2_reg_coef * l2_reg

        # reconstruction loss
        xp_losses = list()
        xp_losses.append(self.compute_pred_loss(inputs, xp_list[0]))

        for i_xp in range(1, len(xp_list)):
            xp_losses.append(self.compute_pred_loss(
                h_list[i_xp - 1], xp_list[i_xp]))

        # output loss
        loss = self.compute_output_loss(y, labels) + l2_reg

        return AutoencoderNetConf(inputs, labels, y, loss, xp_list, h_list, xp_losses)

    def get_pred_optimizers(self, NetConf, learning_rate=0.0002, beta1=0.5):
        opt_list = list()

        for l in range(len(NetConf.xp_losses)):
            xp_l = NetConf.xp_losses[l]

            opt_list.append(
                tf.train.AdamOptimizer(
                    learning_rate=learning_rate, beta1=beta1
                ).minimize(xp_l, var_list=self.full_layers[l].params)
            )

        return opt_list
