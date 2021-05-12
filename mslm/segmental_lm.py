"""
PyTorch modules for constructing a Segmental Language Model (Sun and Deng 2018;
Kawakami, Dyer, and Blunsom 2019), used for character modeling and unsupervised
segmentation

Authors:
    C.M. Downey (cmdowney@uw.edu)
"""
import math
import time
import warnings
from typing import List, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.nn.modules.transformer import (
    TransformerEncoder, TransformerEncoderLayer
)

from .lattice import AcyclicLattice
from .segmental_transformer import (
    SegmentalTransformerEncoder
)


class SegmentalLanguageModel(nn.Module):
    """
    A PyTorch implementation of a Segmental Language Model (Sun and Deng 2018;
    Kawakami, Dyer, and Blunsom 2019; Downey et al. 2021)

    A Segmental Language Model consists of a "Context Encoder", which encodes a
    language-modeling context with which to predict the next or missing segment,
    depending on the modeling task, a "Segment Decoder" which takes the context
    encoding as a start symbol and generates a probability distribution over the
    possible next segments, and optionally a "Lexicon" which can be used as an
    alternative generator to the Segment Decoder (See Kawakami, Dyer, and
    Blunsom for details). The probabilities taken from the decoder are used to
    define all possible segmentations over an input sequence, and regress on the
    marginal likelihood. There is also an additional option to penalize based on
    the expected length of segments, preventing the model from under-segmenting
    """
    def __init__(self, parameters):
        super().__init__()
        for key in parameters:
            setattr(self, key, parameters[key])

        # If `pretrained_embedding` is not null, import the embedding from the
        # NumPy data, otherwise randomly initialize the embedding layer
        if self.pretrained_embedding is not None:
            shard_embedding = nn.Parameter(
                torch.from_numpy(self.pretrained_embedding).float()
            )
        else:
            shard_embedding = torch.zeros(self.vocab_size, self.model_dim)
            nn.init.uniform_(shard_embedding, a=-0.1, b=0.1)

        self.embedding = nn.Embedding.from_pretrained(
            shard_embedding, freeze=False
        )

        # Initialize the encoder from the model parameters
        self.encoder = SLMEncoder(
            self.model_dim,
            self.encoder_dim,
            self.num_enc_layers,
            enc_type=self.enc_type,
            input_dropout=self.model_dropout,
            encoder_dropout=self.encoder_dropout,
            num_heads=self.num_heads,
            ffwd_dim=self.ffwd_dim,
            autoencoder=self.autoencoder,
            attention_window=self.attention_window,
            smart_position=self.smart_position
        )

        # Initialize the lexicon if it is being used
        if self.use_lexicon:
            self.lexicon = SLMLexicon(
                self.encoder_dim, self.model_dim, self.subword_vocab_size
            )
        else:
            self.lexicon = None

        # Initialize the linear layers that project the context encodings to the
        # start symbols and h values of the decoder
        self.encoding_to_start_symbol = nn.Linear(
            self.encoder_dim, self.model_dim
        )
        self.encoding_to_h = nn.Linear(self.encoder_dim, self.model_dim)

        # Initialize the decoder from the model parameters
        self.decoder = SLMDecoder(
            self.model_dim,
            self.num_dec_layers,
            self.vocab_size,
            dropout=self.model_dropout
        )

        initrange = 0.1
        self.encoding_to_start_symbol.weight.data.uniform_(
            -initrange, initrange
        )
        self.encoding_to_start_symbol.bias.data.zero_()
        self.encoding_to_h.weight.data.uniform_(-initrange, initrange)
        self.encoding_to_h.bias.data.zero_()

    def forward(
        self,
        data: Tensor,
        lengths: List[int],
        max_seg_len: int,
        eoseg_index: int,
        chars_to_subword_id: dict = None,
        length_exponent: int = 2,
        length_penalty_lambda: float = None
    ):
        """
        Run a forward pass of the Segmental Language Model, calculating
        probabilities for each possible segment, and returning the normalized
        marginal likelihood of each sequence as loss. Also return the most
        probable segmentation

        Args:
            data: The batch of input sequences
            lengths: The list of the real (unpadded) lengths for each sequence
            max_seg_len: The maximum segment length to be encoded
            eoseg_index: The index of the <eoseg> (end-of-segment) character
            chars_to_subword_id: A mapping of character-index tuples to the
                indices of the subword/segment they constitute. Default:
                ``None``
            length_exponent: The exponent to which to raise each segment length
                if expected length regularization is being used. Default: 2
            length_penalty_lambda: The lambda used to control the strength of
                the expected length penalty. If ``None``, expected length will
                not be calculated at all. Default: ``None``
        """

        seq_length = data.size(0)
        batch_size = data.size(1)
        assert len(lengths) == batch_size

        # Get the device for the model
        using_cuda = next(self.parameters()).is_cuda
        device = torch.device('cuda' if using_cuda else 'cpu')

        max_seg_len = min(max_seg_len, seq_length - 2)
        
        # Initialize the score for each possible segment to an approximation of
        # infinity (log probability)
        loginf = 1000000.0
        segment_scores = torch.empty(
            seq_length - 1, max_seg_len, batch_size
        ).fill_(-loginf).to(device)
        total_loss = 0

        # If expected length is to be calculated, ensure that the length
        # exponent is set to a valid value
        calculate_length_expectation = False
        if length_penalty_lambda:
            if not length_exponent:
                warnings.warn(
                    "Length penalty lambda is set with a length"
                    " exponent of ``None``. Calculating expectation with an"
                    " exponent of 2"
                )
                length_exponent = 2
            elif length_exponent == 1:
                warnings.warn(
                    "Length penalty lambda is set with a length"
                    " exponent of 1. When using base lengths, the expected"
                    " length of any path through a sequence will be the same."
                    " calculating expectation with an exponent of 1.1"
                )
                length_exponent = 1.1
            calculate_length_expectation = True

        # An efficient way to calculate the segment probabilities is to split
        # the input batch into ranges of start positions that can start a
        # segment of up to length k, down to the range that can only start
        # segments of length 1 (based on the max length of the batch). In each
        # range, the probabilities for segments of length 1 to k will be
        # computed as a batch
        range_by_length = {}
        for seg_len in range(max_seg_len, 0, -1):
            last_index = seq_length - seg_len
            if seg_len == max_seg_len:
                first_index = 0
            else:
                first_index = range_by_length[seg_len + 1][1]
            range_by_length[seg_len] = (first_index, last_index)

        nn_start_time = time.time()

        # Embed the input batch
        embedded_seq = self.embedding(data)

        # Pass the embedded sequences through the Context Encoder
        encoder_output = self.encoder(
            embedded_seq,
            lengths,
            device,
            mask_type=self.mask_type,
            max_seg_len=max_seg_len
        )
        
        # Use the context encodings to create start symbols and initial hidden
        # states for the segment decoder, as well as the lexicon probabilities,
        # if it is being used
        all_start_symbols = torch.tanh(
            self.encoding_to_start_symbol(encoder_output)
        )
        all_h = torch.tanh(self.encoding_to_h(encoder_output))
        if self.use_lexicon:
            all_subword_probs, all_lex_proportions = self.lexicon(
                encoder_output
            )

        # Loop from the minimum to maximum segment length, as well as each
        # index in the sequence from which a segment of that length can begin
        for seg_len in range(max_seg_len, 0, -1):

            # Trim the encoder output down to the number of positions that can
            # actually start a segment of length seg_len
            seg_range = range_by_length[seg_len]
            range_start = seg_range[0]
            range_end = seg_range[1]
            num_seg_starts = range_end - range_start

            start_symbols = all_start_symbols[range_start:range_end, :, :]
            h = all_h[range_start:range_end, :, :]

            if self.use_lexicon:
                # Get the subword logits for the current range and convert them
                # probabilities with softmax. Shape is
                # (num_seg_starts, batch_size, subword_vocab_size)
                subword_logits = all_subword_probs[range_start:range_end, :, :]
                base_logit_size = subword_logits.size(-1)
                # Also get the lexical proportions for this range. Shape is
                # (num_seg_starts, batch_size)
                lexical_proportions = all_lex_proportions[
                    range_start:range_end, :]

                # Create the matrix of subword ids with which to index the
                # logits, adding an additional index at the end for OOV
                # subwords
                subword_ids = np.zeros((num_seg_starts, batch_size, seg_len))
                for i in range(num_seg_starts):
                    real_i = i - range_start
                    for j in range(batch_size):
                        for k in range(seg_len):
                            seg = data[real_i + 1:real_i + k + 2, j].tolist()
                            seg = tuple(seg)
                            if seg in chars_to_subword_id:
                                subword_ids[i, j, k] = chars_to_subword_id[seg]
                            else:
                                subword_ids[i, j, k] = base_logit_size
                subword_ids = (
                    torch.tensor(subword_ids, dtype=torch.int64).to(device)
                )

                # Add negative infinity probs to the end of the logits to allow
                # OOV subwords to not be a problem for torch.gather
                oov_probs = torch.empty(
                    (num_seg_starts, batch_size, 1)
                ).fill_(-loginf).to(device)
                subword_logits = torch.cat((subword_logits, oov_probs), dim=2)

                # Gather the scores for the relevant subwords
                subword_losses = torch.gather(
                    subword_logits, dim=2, index=subword_ids
                )

                # Transpose the subword losses to be
                # (seg_len, num_seg_starts, batch_size), allowing it to be added
                # to the character scores later on
                subword_losses = (
                    subword_losses.transpose(1, 2).transpose(0, 1)
                )

                # Repeat the lexical mixture proportion to be the same for all
                # segment lengths starting at the same position. It is now shape
                # (seg_len, num_seg_starts, batch_size)
                lexical_proportions = lexical_proportions.repeat(seg_len, 1, 1)
                # Set the lexical mixture proportion to 0 where the given
                # subword is OOV
                zeroed_proportions = torch.zeros_like(lexical_proportions)
                lexical_proportions = torch.where(
                    subword_losses > -loginf, lexical_proportions,
                    zeroed_proportions
                )
                # Set the character mixture proportion as 1 minus the lexical
                # proportion
                character_proportions = 1 - lexical_proportions

            # Create a matrix of the 'masked' (original) segments to be fed to
            # the decoder, as well as the matrix of targets to compare against.
            # The matrix of the masked segments is of shape
            # [seg_len, num_segs, batch_size, embedding_size], whereas the
            # matrix of targets is of size [seg_len, num_segs, batch_size]
            # since the targets to compare against are not embedded
            all_masked = []
            all_target = []
            for start_idx in range(range_start + 1, range_end + 1):
                masked_seg = embedded_seq[start_idx:start_idx + seg_len, :, :]
                target_seg = data[start_idx:start_idx + seg_len, :]
                all_masked.append(masked_seg)
                all_target.append(target_seg)

            all_masked = torch.stack(all_masked, dim=1)
            all_target = torch.stack(all_target, dim=1)

            # Aggregate the dimensions for num_segments and batch_size, to
            # conform to the three-dimension standard for the decoder.
            # start_symbols, h, c, and all_masked are now all of size
            # [seg_len, batch_size, embedding_size]. The new 'batch size' is
            # the number of segments times the original batch size. For the
            # start symbols, h, and c, the 'segment length' is simply 1 since
            # they consitute one symbol in the segment
            start_symbols = start_symbols.unsqueeze(0).view(
                1, -1, self.model_dim
            )
            h = h.unsqueeze(0).view(1, -1, self.model_dim)
            h = h.repeat(self.num_dec_layers, 1, 1).contiguous()
            c = torch.zeros_like(h)
            all_masked = all_masked.view(seg_len, -1, self.model_dim)

            # Concatenate the masked segments after the start symbols to be
            # used as the input to the decoder
            decoder_input = torch.cat((start_symbols, all_masked), 0)

            # Run combined encoder output through the LSTM decoder layer
            decoder_output = self.decoder(decoder_input, (h, c))

            # Calculate the probability distribution over each segment using log
            # softmax over the logits for each generated position, then gather
            # the probabilities for the target characters from the distribution
            decoder_output = decoder_output.view(
                seg_len + 1, -1, batch_size, self.vocab_size
            )
            model_probs = f.log_softmax(decoder_output, dim=3)
            target_probs = model_probs[:-1, :, :, :].gather(
                dim=3, index=all_target.unsqueeze(-1)
            ).squeeze(-1)

            # This block is based on the implementation by Sun and Deng (2018).
            # Efficiently calculate the probability for the segment of each
            # length starting at each position, and ending with the <eoseg>.
            # If using the lexicon, properly combine the lexical probability
            # with the character generation probability (not included in Sun and
            # Deng)
            tmp_probs = torch.zeros_like(target_probs[0, :, :])
            for k in range(seg_len):
                tmp_probs = tmp_probs + target_probs[k, :, :]
                eoseg_prob = model_probs[k + 1, :, :, eoseg_index]
                seg_probs = tmp_probs + eoseg_prob
                if self.use_lexicon:
                    seg_probs = seg_probs * character_proportions[k, :, :]
                    sw_probs = (
                        subword_losses[k, :, :] * lexical_proportions[k, :, :]
                    )
                    seg_probs = seg_probs + sw_probs
                segment_scores[range_start:range_end, k, :] = seg_probs

        nn_time = time.time() - nn_start_time
        lattice_start_time = time.time()

        # For the sake of decoding, the sequence length will be the length
        # including <eos> but not <bos>.
        lengths_wo_bos = [length - 1 for length in lengths]
        lengths_wo_bos_tensor = torch.tensor(lengths_wo_bos).to(device)

        # Ensure that <eos> can only ever be generated on its own, and not as
        # part of another segment
        for seq, seq_len in enumerate(lengths_wo_bos):
            for k in range(1, max_seg_len + 1):
                if k == 1:
                    segment_scores[seq_len - k, k:, seq] = -loginf
                else:
                    segment_scores[seq_len - k, k - 1:, seq] = -loginf

        # Form an acyclic/directional lattice over the batch sequences using
        # the calculated segment scores
        lattice = AcyclicLattice(
            arcs=segment_scores,
            lengths=lengths_wo_bos,
            expected_length_exponent=length_exponent
        )

        # Get the batch marginals and best paths from the lattice, normalizing
        # the marginals by sequence length
        marginals = -lattice.marginal()
        marginals = marginals / lengths_wo_bos_tensor
        best_paths, _ = lattice.best_path()

        time_profile = {}
        time_profile['nn'] = nn_time

        # If a length expectation penalty is being computed, combine it with the
        # main (bpc) loss, then return both the penalized and unpenalized loss,
        # along with the best paths. Otherwise, just return the main loss and
        # the best paths
        if calculate_length_expectation:
            length_expectations = lattice.expected_length()
            length_expectations = length_expectations / lengths_wo_bos_tensor
            total_losses = (
                marginals + (length_penalty_lambda * length_expectations)
            )
            total_loss = total_losses.mean()
            bpc_loss = marginals.mean().item()

            lattice_time = time.time() - lattice_start_time
            time_profile['lattice'] = lattice_time

            return total_loss, bpc_loss, best_paths, time_profile
        else:
            total_loss = marginals.mean()

            lattice_time = time.time() - lattice_start_time
            time_profile['lattice'] = lattice_time

            return total_loss, best_paths, time_profile


class SLMEncoder(nn.Module):
    """
    The "Context Encoder" component of the Segmental Language Model. Encodes a
    language modeling context as a hidden state, either via an LSTM or Segmental
    Transformer

    Args:
        embedding_dim: The dimension of the input embeddings
        encoder_dim: The dimension of the hidden states and output of the
            encoder
        num_enc_layers: The number of layers to include in the encoder
        enc_type: The architecture of encoder to use. Current options are
            ``lstm`` and ``transformer``. Default: ``transformer``
        input_dropout: The rate of dropout to apply before the encoder
            (after positional encoding). Default: 0.1
        encoder_dropout: The rate of dropout to apply between encoder layers.
            Default: 0.1
        num_heads: The number of attention heads to use for a transformer
            encoder. Default: 4
        ffwd_dim: The dimension of the feedforward layers in a transformer
            encoder. Default: 256
        autoencoder: Whether to apply no attention mask and use an autoencoding
            setup rather than a traditional language model. Default: ``False``
        attention_window: The size of the attention window to apply around or
            before the masked/unknown span
        max_seq_length: The absolute max sequence length expected to be encoded
            (used for sinusoidal positional encoding). Default: 4096
        smart_position: Whether to learn the proportion with which to add the 
            original and positional embeddings at each position. Adds a linear
            layer with ``2 * encoder_dim`` parameters. Default: ``False``
    """
    def __init__(
        self,
        embedding_dim: int,
        encoder_dim: int,
        num_enc_layers: int,
        enc_type: str = 'transformer',
        input_dropout: float = 0.1,
        encoder_dropout: float = 0.1,
        num_heads: int = 4,
        ffwd_dim: int = 256,
        autoencoder: bool = 'False',
        attention_window: int = None,
        max_seq_length: int = 2048,
        smart_position: bool = False
    ):
        super().__init__()
        self.enc_type = enc_type
        self.autoencoder = autoencoder
        self.attention_window = attention_window

        if self.attention_window:
            warnings.warn(
                "Class SLMEncoder is set with a non-null value for"
                " ``attention_window``. Some mask interactions in PyTorch\'s"
                " transformer implementation, including adding an attention"
                " window (also known as \"bucket attention\") can cause NaN"
                " gradients or more silent errors. This happens when any row of"
                " the self-attention matrix is inadvertently completely masked"
                " off. In this case, you need to ensure that the max number of"
                " pad tokens on any sequence is strictly less than the length"
                " of the window on either side"
            )

        self.input_dropout = nn.Dropout(p=input_dropout)
        self.input_to_enc_dim = nn.Linear(embedding_dim, encoder_dim)

        # If the encoder is to be transformer-based, initialize the Positional
        # Encoding component and set the h and c state to None
        if self.enc_type == 'transformer':
            
            # Register a static sinusoidal positional encoding, as well as an
            # optional feedforward layer to determine the relative strength of
            # the original and positional embeddings
            pe = np.zeros((max_seq_length, encoder_dim))
            for pos in range(max_seq_length):
                for i in range(0, math.ceil(encoder_dim / 2)):
                    pe[pos, 2 * i] = np.sin(pos / (10000 ** (2 * i / encoder_dim)))
                    if 2 * i + 1 < encoder_dim:
                        pe[pos, 2 * i + 1] = np.cos(pos / (10000 ** (2 * i / encoder_dim)))
            model_dtype = self.input_to_enc_dim.weight.dtype
            pe = torch.tensor(pe, dtype=model_dtype).unsqueeze(1)
            self.register_buffer('pe', pe)
            if smart_position:
                self.positional_emb_proportion = nn.Linear(2 * encoder_dim, 1, bias=False)
                self.positional_emb_proportion.weight.data.fill_(0.0001)
                self.emb_scale_factor = None
            else:
                self.positional_emb_proportion = None
                self.emb_scale_factor = nn.Parameter(torch.tensor([1.0]))
            
            self.h_init_state = None
            self.c_init_state = None
            
            # A transformer-based encoder has the additional option of being run
            # as an autoencoder (i.e. having no positions masked out). This is
            # not a language model in the traditional sense
            if self.autoencoder:
                encoder_layer = TransformerEncoderLayer(
                    encoder_dim,
                    num_heads,
                    dim_feedforward=ffwd_dim,
                    dropout=encoder_dropout
                )
                norm = nn.LayerNorm(encoder_dim)
                self.encoder = TransformerEncoder(
                    encoder_layer, num_enc_layers, norm=norm
                )
            else:
                self.encoder = SegmentalTransformerEncoder(
                    encoder_dim,
                    num_heads,
                    num_enc_layers,
                    ffwd_dim=ffwd_dim,
                    dropout=encoder_dropout
                )
        
        # If the encoder is to be lstm-based, initialize the h and c initial
        # states, also set the positional encoding to be None
        elif self.enc_type == 'lstm':
            self.pos_encoder = None
            self.positional_emb_proportion = None
            self.emb_scale_factor = None
            self.h_init_state = nn.Parameter(
                torch.zeros(num_enc_layers, 1, encoder_dim)
            )
            self.c_init_state = nn.Parameter(
                torch.zeros(num_enc_layers, 1, encoder_dim)
            )
            self.encoder = nn.LSTM(
                encoder_dim,
                encoder_dim,
                num_layers=num_enc_layers,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f'Encoder type {self.enc_type} is not valid')

    def forward(
        self,
        x: Tensor,
        lengths: List[int],
        device,
        mask_type: str = 'cloze',
        max_seg_len: int = 1
    ) -> Tensor:
        """
        Encode the language modeling context for each position in the input
        sequences

        Args:
            x: The embedded input batch of sequences
            lengths: The list of the real (unpadded) lengths for each sequence
            device: The PyTorch device to which to move the mask
            mask_type: The type of attention mask to use with a transformer
                encoder. Default: ``cloze``
            max_seg_len: The maximum segment length to be encoded. Default: 1
        """
        seq_len = x.size(0)
        batch_size = x.size(1)

        x = self.input_to_enc_dim(x)

        if self.enc_type == 'transformer':
            # Create the padding mask
            padding_mask = torch.zeros(batch_size, seq_len)
            for seq in range(batch_size):
                if lengths[seq] < seq_len:
                    for pad in range(lengths[seq], seq_len):
                        padding_mask[seq][pad] = 1
            padding_mask = padding_mask.bool().to(device)

            # Add the positional encoding to the embedding
            pos_encoding = self.pe[:seq_len, :]
            if self.positional_emb_proportion:
                pos_encoding = pos_encoding.repeat(1, batch_size, 1)
                concatenated_embedding = torch.cat((x, pos_encoding), dim=2)
                emb_factor = 1.0 + torch.relu(
                    self.positional_emb_proportion(concatenated_embedding)
                )
                scaled_embedding = emb_factor * x
                pos_embedded_seq = scaled_embedding + pos_encoding
            else:
                pos_embedded_seq = (self.emb_scale_factor * x) + pos_encoding

            # Apply dropout before feeding to the encoder
            pos_embedded_seq = self.input_dropout(pos_embedded_seq)

            # Run the input through the transformer block
            if not self.autoencoder:
                attn_mask = self.encoder.get_mask(
                    seq_len,
                    seg_len=max_seg_len,
                    shape=mask_type,
                    window=self.attention_window
                ).to(device)
                encoder_output = self.encoder(
                    pos_embedded_seq,
                    attn_mask=attn_mask,
                    padding_mask=padding_mask
                )
            else:
                encoder_output = self.encoder(
                    pos_embedded_seq, src_key_padding_mask=padding_mask
                )
        elif self.enc_type == 'lstm':
            # Apply dropout before feeding to the encoder
            x = self.input_dropout(x)
            # Expand h and c to match the batch size, and run the input through
            # the LSTM block
            h = self.h_init_state.expand(-1, batch_size, -1).contiguous()
            c = self.c_init_state.expand(-1, batch_size, -1).contiguous()
            rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
            encoder_output, _ = self.encoder(rnn_input, (h, c))
            encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output)
        else:
            raise ValueError(f'Encoder type {self.enc_type} is not valid')

        return encoder_output


class SLMLexicon(nn.Module):
    """
    The optional "Lexicon" or "Memory" component of the Segmental Language
    Model. Decodes context/position encodings to logits over a segmental
    vocabulary, as well as a mixture proportion for combining this loss with the
    character-generation loss

    Args:
        d_enc: The dimension of the encodings returned from the encoder
        d_model: The dimension of the hidden states used in the decoder and the
            rest of the model
        subword_vocab_size: The size of the vocabulary over subwords/segments
        initrange: The positive end of the initialization range for the lexicon
            layers. Default: 0.1
    """
    def __init__(
        self,
        d_enc: int,
        d_model: int,
        subword_vocab_size: int,
        initrange: float = 0.1
    ):
        super().__init__()
        self.encoding_to_subword_hidden = nn.Linear(d_enc, d_model)
        self.subword_decoder = nn.Linear(d_model, subword_vocab_size)
        self.encoding_to_mixture_hidden = nn.Linear(d_enc, d_model)
        self.hidden_to_mixture_proportion = nn.Linear(d_model, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.encoding_to_subword_hidden.weight.data.uniform_(
            -initrange, initrange
        )
        self.subword_decoder.weight.data.uniform_(-initrange, initrange)
        self.encoding_to_mixture_hidden.weight.data.uniform_(
            -initrange, initrange
        )
        self.hidden_to_mixture_proportion.weight.data.uniform_(
            -initrange, initrange
        )
        self.encoding_to_subword_hidden.bias.data.zero_()
        self.subword_decoder.bias.data.zero_()
        self.encoding_to_mixture_hidden.bias.data.zero_()

    def forward(self, encodings: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decode the segment encodings to logits over the subword vocabulary and
        mixture proportions for the Lexicon

        Args:
            encodings: The context/positional encodings output by the SLM
                Encoder
        """
        subword_encodings = self.encoding_to_subword_hidden(encodings)
        subword_scores = self.subword_decoder(subword_encodings)
        subword_probs = self.log_softmax(subword_scores)

        mixture_encodings = self.encoding_to_mixture_hidden(encodings)
        mixture_outputs = self.hidden_to_mixture_proportion(mixture_encodings)
        mixture_proportions = self.sigmoid(mixture_outputs.squeeze(-1))

        return subword_probs, mixture_proportions


class SLMDecoder(nn.Module):
    """
    The "Segment Decoder" component of the Segmental Language Model. Decodes a
    probability distribution over possible next segments based on the "context"
    encoding from the SLM Encoder

    Args:
        d_model: The dimension of the context encoding and output hidden states
        num_dec_layers: The number of lstm layers to include in the decoder
        vocab_size: The size of the character/symbol vocab for projection to
            logits
        dropout: The rate of dropout applied to inputs to the decoder. Default:
            0.1
        initrange: The positive end of the initialization range for the linear
            layer. Default: 0.1
    """
    def __init__(
        self,
        d_model: int,
        num_dec_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
        initrange: float = 0.1
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(p=dropout)
        self.segment_lstm = nn.LSTM(
            d_model, d_model, num_layers=num_dec_layers, dropout=dropout
        )
        self.hidden_to_vocab = nn.Linear(d_model, vocab_size)
        self.hidden_to_vocab.weight.data.uniform_(-initrange, initrange)
        self.hidden_to_vocab.bias.data.zero_()

    def forward(self, x: Tensor, inits: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Decode to logits over the next segments based on the input context
        encodings and characters

        Args:
            x: The input context/positional encoding plus the next characters
                of each segment
            inits: The initial hidden and cell states for the lstm
        """
        x = self.input_dropout(x)
        decoder_output, _ = self.segment_lstm(x, inits)
        decoder_output = self.hidden_to_vocab(decoder_output)
        return decoder_output
