"""
A custom PyTorch Transformer encoder for doing cloze-type prediction over spans
of positions (segments), using the additive attention mask rather than
converting tokens to special ``<mask>`` tokens

Authors:
    C.M. Downey (cmdowney@uw.edu), PyTorch (see ``PositionalEncoding``)
"""
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.transformer import (TransformerEncoderLayer, _get_clones)


class SegmentalTransformerEncoder(nn.Module):
    """
    A Transformer encoder for doing segmental cloze predictions over spans of
    masked positions

    Args:
        d_model: The input and output dimension of the encoder
        n_head: The number of attention heads
        n_layers: The number of encoder layers in the block
        ffwd_dim: The dimension of the two feedforward layers within each
            Transformer encoder layer. Default: 256
        dropout: The rate of dropout in the encoder. Default: 0.1
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_layers: int,
        ffwd_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ffwd_dim = ffwd_dim
        self.primary_encoder = InitialSpanEncoder(
            d_model, n_head, dim_feedforward=self.ffwd_dim, dropout=dropout
        )
        self.n_layers = n_layers - 1
        if self.n_layers > 0:
            subsequent_encoder = SubsequentSpanEncoder(
                d_model, n_head, dim_feedforward=self.ffwd_dim, dropout=dropout
            )
            self.subsequent_layers = _get_clones(
                subsequent_encoder, self.n_layers
            )
        else:
            self.subsequent_layers = None
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        attn_mask: Tensor = None,
        padding_mask: Tensor = None
    ) -> Tensor:
        """
        Encode input with the Segmental Transformer Encoder

        Args:
            src: The input sequence to encode
            attn_mask: The additive attention mask with which to mask out the
                span encoded at each position. Default: ``None``
            padding_mask: The mask for the padded positions of each key.
                Default: ``None``
        Shape:
            - src: ``(S, B, E)``
            - attn_mask: ``(S, S)``
            - padding_mask: ``(S, B)``
            - output: ``(S, B, E)``
            where ``S`` is the src sequence length, ``B`` is the batch size, 
            and ``E`` is the embedding/model dimension
        """
        output = self.primary_encoder(
            src, attn_mask=attn_mask, padding_mask=padding_mask
        )
        for i in range(self.n_layers):
            output = self.subsequent_layers[i](
                output, src, attn_mask=attn_mask, padding_mask=padding_mask
            )
        if self.norm:
            output = self.norm(output)
        return output

    @staticmethod
    def get_mask(
        seq_len: int,
        shape: str = 'cloze',
        seg_len: int = None,
        window: int = None
    ) -> Tensor:
        """
        Generate the proper attention mask for use with the Segmental
        Transformer Encoder, using either a Cloze or Causal/Subsequent modeling
        assumption

        Args:
            seq_len: The sequence length for the input
            shape: The mask shape/type. If ``cloze``, predicts a masked segment
                based on a bidirectional context. If ``subsequent``, predicts a
                segment based on its leftward context. Default: ``cloze``
            seg_len: The maximum segment length to be masked and predicted.
                Default: ``None``
            window: The size of the attention window with which to predict the
                masked segment. If the mask shape is ``cloze`` and the window
                size is ``n``, this means ``n/2`` unmasked positions on either
                side of the segment. If the mask shape is ``subsequent``, this
                means ``n`` unmasked positions to the left of the segment.
                Default: ``None``
        Returns:
            An attention mask for use in the Segmental Transformer Encoder
        """
        if shape == 'cloze':
            if window:
                window = window // 2
            mask = (np.ones((seq_len, seq_len))) == 1
            for i in range(seq_len):
                for j in range(1, min(seg_len + 1, seq_len - i)):
                    mask[i, i + j] = False
                if window:
                    for k in range(window, i + 1):
                        mask[i, i - k] = False
                    for k in range(seg_len + window + 1, seq_len - i):
                        mask[i, i + k] = False
        elif shape == 'subsequent':
            mask = (np.triu(np.ones((seq_len, seq_len))) == 1).transpose()
            if window:
                for i in range(seq_len):
                    for k in range(window, i + 1):
                        mask[i, i - k] = False
        else:
            raise TypeError(f"Transformer mask shape {shape} is not recognized")

        mask = torch.tensor(mask)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask


class InitialSpanEncoder(TransformerEncoderLayer):
    """
    The initial layer for the Segmental Transformer Encoder. Representations of
    the source sequence attend over all unmasked positions in the sequence

    The encoding at position ``i`` represents the masked span starting at
    position ``i+1``
    
    Args:
        src: The input sequence to encode
        attn_mask: The additive attention mask with which to mask out the
            span encoded at each position. Default: ``None``
        padding_mask: The mask for the padded positions of each key.
            Default: ``None``
    """
    def forward(
        self,
        src: Tensor,
        attn_mask: Tensor = None,
        padding_mask: Tensor = None
    ) -> Tensor:
        src1 = self.self_attn(
            src, src, src, attn_mask=attn_mask, key_padding_mask=padding_mask
        )[0]
        src = self.norm1(self.dropout1(src1))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class SubsequentSpanEncoder(TransformerEncoderLayer):
    """
    The subsequent layers for the Segmental Transformer Encoder. The encoded
    representations from previous layers attend over all unmasked positions of
    the original source sequence (to prevent information leaks from "under" the
    mask)

    The encoding at position ``i`` represents the masked span starting at
    position ``i+1``
    
    Args:
        enc: The encoded representation from previous segmental encoder layers
        src: The original input sequence to encode
        attn_mask: The additive attention mask with which to mask out the
            span encoded at each position. Default: ``None``
        padding_mask: The mask for the padded positions of each key.
            Default: ``None``
    """
    def forward(
        self,
        enc: Tensor,
        src: Tensor,
        attn_mask: Tensor = None,
        padding_mask: Tensor = None
    ) -> Tensor:
        enc1 = self.self_attn(
            enc, src, src, attn_mask=attn_mask, key_padding_mask=padding_mask
        )[0]
        enc = self.norm1(enc + self.dropout1(enc1))
        enc2 = self.linear2(self.dropout(self.activation(self.linear1(enc))))
        enc = self.norm2(enc + self.dropout2(enc2))
        return src


class PositionalEncoding(nn.Module):
    """
    Static sinusoidal positional embeddings to be added to the input to a
    transformer

    (From https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

    Args:
        d_model: The dimension of the embeddings
        dropout: The rate of dropout after the encodings are added
        max_len: The maximum expected sequence length the model will ever encode
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
