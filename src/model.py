# codeing: utf-8

import math
import os
import numpy as np

import mxnet as mx
from mxnet import gluon, autograd as ag
from mxnet.gluon import nn, rnn
from mxnet.gluon.loss import L2Loss

from mxnet.gluon.block import HybridBlock, Block
from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, MultiHeadAttentionCell


class QuestionAnsweringClassifier(HybridBlock):
    """
    Your primary model block for Bi-LSTM and attention model
    """

    def __init__(self, emb_input_dim, emb_output_dim, max_seq_len=32, num_classes=19,
                 num_layers=6, dropout=.2, attn_cell='multi_head'):
        super(QuestionAnsweringClassifier, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
            self.bilstm = rnn.LSTM(hidden_size=2048,
                                   dropout=dropout,
                                   bidirectional=True,
                                   # input_size=?,
                                   )
            self.attention_transform = BaseEncoder(attention_cell=attn_cell,
                                                   num_layers=num_layers,
                                                   units=emb_output_dim,
                                                   hidden_size=2048,
                                                   max_length=max_seq_len,
                                                   num_heads=4,
                                                   scaled=True,
                                                   dropout=dropout,
                                                   use_residual=True,
                                                   output_attention=False,
                                                   weight_initializer=None,
                                                   bias_initializer='zeros',
                                                   prefix=None,
                                                   params=None)
            self.output = nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, data, indices):
        """
        Inputs:
         - data The sentence representation (token indices to feed to embedding layer)
         - inds A vector - shape (2,) of two indices referring to positions of the two arguments
        NOTE: Your implementation may involve a different approach
        """
        embedded = self.embedding(data)  ## shape (batch_size, length, emb_dim)
        after_attn = self.attention_transform(embedded)  # shape ()
        outputs = self.output(after_attn)
        return outputs


class PositionwiseFFN(HybridBlock):
    """
    Taken from the gluon-nlp library.
    """

    def __init__(self, units=512, hidden_size=1024, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  activation=activation,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        outputs = self.ffn_1(inputs)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


def _get_attention_cell(attention_cell, units=None,
                        scaled=True, num_heads=None,
                        use_bias=False, dropout=0.0):
    """
    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell), \
            f'attention_cell must be either string or AttentionCell. Received attention_cell={attention_cell}'
        return attention_cell


class BaseEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """

    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BaseEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        arg_inputs: Symbol or NDArray
            Input arguments. Shape (batch_size, 2)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the encoder cell. Shape (batch_size, 2, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        # outputs has shape (batch_size, max_seq_len, units)
        # attention_weights has shape  (batch_size, 4, max_seq_len, max_seq_len)
        outputs, attention_weights = self.attention_cell(inputs, inputs, None, None)
        outputs = self.proj(outputs)  # shape (batch_size, max_seq_len, units)
        outputs = self.dropout_layer(outputs)  # shape (batch_size, max_seq_len, units)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        return outputs


def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


class BaseEncoder(HybridBlock):

    def __init__(self, attention_cell='multi_head', num_layers=6,
                 units=512, hidden_size=2048, max_length=64,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BaseEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, \
            'In TransformerEncoder, The units should be divided exactly ' \
                f'by the number of heads. Received units={units}, num_heads={num_heads}'
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            ## !!! Original code creates a number of attention layers
            ## !!! Hard-coded here for a single base encoder cell for simplicity
            self.transformer_cells = nn.HybridSequential()
            with self.transformer_cells.name_scope():
                for i in range(num_layers):
                    self.transformer_cells.add(BaseEncoderCell(units=units,
                                                               hidden_size=hidden_size,
                                                               num_heads=num_heads,
                                                               attention_cell=attention_cell,
                                                               weight_initializer=weight_initializer,
                                                               bias_initializer=bias_initializer,
                                                               dropout=dropout,
                                                               use_residual=use_residual,
                                                               scaled=scaled,
                                                               output_attention=output_attention,
                                                               prefix=f'transformer{i}'))

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(BaseEncoder, self).__call__(inputs)

    def hybrid_forward(self, F, inputs, position_weight):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        arg_pos: int array pair

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        """
        batch_size = inputs.shape[0]
        steps = F.arange(self._max_length)
        inputs = F.broadcast_add(inputs,
                                 F.expand_dims(F.Embedding(steps,
                                                           position_weight,
                                                           self._max_length,
                                                           self._units),
                                               axis=0))
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = self.transformer_cells(inputs)
        return outputs
