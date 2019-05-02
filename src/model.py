# codeing: utf-8

import math
import os
import numpy as np
import logging

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
            self.bilstm_question = rnn.LSTM(hidden_size=2048,
                                            dropout=dropout,
                                            bidirectional=True,
                                            )
            self.bilstm_context = rnn.LSTM(hidden_size=2048,
                                           dropout=dropout,
                                           bidirectional=True,
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

            # self.attention_transform = BaseEncoderCell(units=128,
            #                                            hidden_size=2048,
            #                                            num_heads=4,
            #                                            attention_cell=attn_cell,
            #                                            weight_initializer=None,
            #                                            bias_initializer='zeros',
            #                                            dropout=dropout,
            #                                            use_residual=False,
            #                                            scaled=True,
            #                                            output_attention=False,
            #                                            prefix=f'transformer')
            self.output = nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, question, context):
        # embedding layers for question and context
        question_embedded = self.embedding(question)  ## shape (batch_size, max_len, emb_dim)
        context_embedded = self.embedding(context)
        logging.debug(f"shape of question after embedding {question_embedded.shape}")
        logging.debug(f"shape of context after embedding {context_embedded.shape}")

        # bilstm layers for question and context
        question_after_lstm = self.bilstm_question(question_embedded)
        context_after_lstm = self.bilstm_context(context_embedded)
        logging.debug(f"shape of question after lstm {question_after_lstm.shape}")
        logging.debug(f"shape of context after lstm {context_after_lstm.shape}")

        # attention layer
        after_attn = self.attention_transform(question_after_lstm, context_after_lstm)
        logging.debug(f"shape of after_attn {after_attn.shape}")

        # output layer
        outputs = self.output(after_attn)
        logging.debug(f"shape of outputs after output layer {outputs.shape}")
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

    def hybrid_forward(self, F, query, key):  # pylint: disable=arguments-differ
        # outputs has shape (batch_size, max_seq_len, units)
        # attention_weights has shape  (batch_size, 4, max_seq_len, max_seq_len)
        outputs, attention_weights = self.attention_cell(query, key, None, None)
        outputs = self.proj(outputs)  # shape (batch_size, max_seq_len, units)
        outputs = self.dropout_layer(outputs)  # shape (batch_size, max_seq_len, units)
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        return outputs


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
            self.transformer_cells = BaseEncoderCell(units=units,
                                                     hidden_size=hidden_size,
                                                     num_heads=num_heads,
                                                     attention_cell=attention_cell,
                                                     weight_initializer=weight_initializer,
                                                     bias_initializer=bias_initializer,
                                                     dropout=dropout,
                                                     use_residual=use_residual,
                                                     scaled=scaled,
                                                     output_attention=output_attention,
                                                     prefix=f'transformer')

    def __call__(self, query, key):  # pylint: disable=arguments-differ
        return super(BaseEncoder, self).__call__(query, key)

    def hybrid_forward(self, F, query, key):  # pylint: disable=arguments-differ
        outputs = self.transformer_cells(query, key)
        return outputs
