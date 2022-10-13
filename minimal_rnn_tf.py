from __future__ import absolute_import
import warnings

# import tensorflow as tensorflow 
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
# from tensorflow.keras.legacy import interfaces
from tensorflow.keras.layers import RNN


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.

    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.

    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.int_shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class MinimalRNNCell(Layer):
    """Minimarl RNN Cell

        # Arguments
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use
                (see [activations](keras/activations.md)).
                If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
                for the recurrent step
                (see [activations](keras/activations.md)).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
                (see [initializers](../initializers.md)).
            bias_initializer: Initializer for the bias vector
                (see [initializers](../initializers.md)).
            unit_forget_bias: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Setting it to true will also force `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            bias_regularizer: Regularizer function applied to the bias vector
                (see [regularizer](../regularizers.md)).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
                (see [regularizer](../regularizers.md)).
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix
                (see [constraints](../constraints.md)).
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix
                (see [constraints](../constraints.md)).
            bias_constraint: Constraint function applied to the bias vector
                (see [constraints](../constraints.md)).
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.

        # References
            - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
            - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
            - [Bahdanau, Cho & Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/pdf/1409.0473.pdf)
            - [MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks](https://arxiv.org/abs/1711.06788v1).]()
        """
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(MinimalRNNCell, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.implementation = implementation
        self.state_spec = [InputSpec(shape=(None, self.units)),]
        self.state_size = (self.units,)


    def build(self, input_shape):
        self.timestep_dim = 1 #input_shape[0]
        self.input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        else:
            self.bias = None
            self.attention_bias = None
            self.attention_recurrent_bias = None

        self.recurrent_kernel_u = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, self.units: self.units * 2]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_u = self.bias[self.units: self.units * 2]
        else:
            self.bias_z = None
            self.bias_u = None
        self.built = True


    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(1)]
        else:
            self._dropout_mask = None


    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(2)]
        else:
            self._recurrent_dropout_mask = None

    def call(self, inputs, states, training=None):
        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_x = inputs * dp_mask[0]
            else:
                inputs_x = inputs

            x_z = K.dot(inputs_x, self.kernel)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.bias_z)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_u = h_tm1 * rec_dp_mask[0]
                h_tm1_z = h_tm1 * rec_dp_mask[1]
            else:
                h_tm1_u = h_tm1
                h_tm1_z = h_tm1

            z = self.activation(x_z)
            u = K.dot(h_tm1_u, self.recurrent_kernel_u) + K.dot(h_tm1_z, self.recurrent_kernel_z)
            if self.use_bias:
                u = K.bias_add(u, self.bias_u)
            u = self.recurrent_activation(u)

        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias_z)
            z = self.activation(z)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            u = K.dot(h_tm1, self.recurrent_kernel_u) + K.dot(h_tm1, self.recurrent_kernel_z)
            if self.use_bias:
                u = K.bias_add(u, self.bias_u)
            u = self.recurrent_activation(u)

        h = u * h_tm1 + (1 - u) * z
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h,]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,}
        base_config = super(MinimalRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MinimalRNN(RNN):
    """Minimarl RNN Cell

        # Arguments
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use
                (see [activations](keras/activations.md)).
                If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
                for the recurrent step
                (see [activations](keras/activations.md)).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
                (see [initializers](../initializers.md)).
            bias_initializer: Initializer for the bias vector
                (see [initializers](../initializers.md)).
            unit_forget_bias: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Setting it to true will also force `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            bias_regularizer: Regularizer function applied to the bias vector
                (see [regularizer](../regularizers.md)).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
                (see [regularizer](../regularizers.md)).
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix
                (see [constraints](../constraints.md)).
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix
                (see [constraints](../constraints.md)).
            bias_constraint: Constraint function applied to the bias vector
                (see [constraints](../constraints.md)).
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.

        # References
            - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
            - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
            - [Bahdanau, Cho & Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/pdf/1409.0473.pdf)
            - [MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks](https://arxiv.org/abs/1711.06788v1).]()
        """
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 return_attention=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')

        if K.backend() == 'cntk':
            if not kwargs.get('unroll') and (dropout > 0 or recurrent_dropout > 0):
                warnings.warn(
                    'RNN dropout is not supported with the CNTK backend '
                    'when using dynamic RNNs (i.e. non-unrolled). '
                    'You can either set `unroll=True`, '
                    'set `dropout` and `recurrent_dropout` to 0, '
                    'or use a different backend.')
                dropout = 0.
                recurrent_dropout = 0.

        cell = MinimalRNNCell(units,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              unit_forget_bias=unit_forget_bias,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              implementation=implementation)
        super(MinimalRNN, self).__init__(cell,
                                         return_sequences=return_sequences,
                                         return_state=return_state,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         unroll=unroll,
                                         **kwargs)
        self.return_attention = return_attention

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        self.cell._generate_dropout_mask(inputs, training=training)
        self.cell._generate_recurrent_dropout_mask(inputs, training=training)
        return super(MinimalRNN, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state,
                                            constants=constants)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def activity_regularizer(self):
        return self.cell.activity_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,}
        base_config = super(MinimalRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return super(MinimalRNN, cls).from_config(config, custom_objects=custom_objects)