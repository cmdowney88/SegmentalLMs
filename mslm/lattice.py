"""
A module for performing regressible graph-lattice computations built on PyTorch
tensors, such as computations of marginal likelihood, best path, and expected
value over a tensor-represented graph

Authors:
    C.M. Downey (cmdowney@uw.edu)
"""
from typing import List

import torch
import numpy as np
from torch import Tensor


def log_expectation(log_probs: Tensor, values: Tensor) -> Tensor:
    """
    Get the numerically-stabilized log expected value over a set of log
    probabilities and their associated values
    """
    a = log_probs.max(dim=0).values
    normalized_log_probs = log_probs - a
    raw_log_expectations = torch.log(
        torch.sum(values * torch.exp(normalized_log_probs))
    )
    normalized_log_expectations = (raw_log_expectations + a)
    return normalized_log_expectations


class AcyclicLattice():
    """
    PyTorch class instantiating an acyclic lattice graph, such as one that
    defines all paths through an ordered sequence. Can be used to return a
    Tensor containing the marginal likelihood of the graph, the expected length
    of a path through the graph, and the decoded most probable path

    Args:
        arcs: A tensor of size ``(max_length, max_segment_length, batch_size)``
            Representing the probability of segments, or equivalently, arcs
            through the ordered graph of the sequence. The matrix is read such
            that arcs[0,2] represents the probability of the segment from
            position 0 to 3, for example (the segment length is the second
            index plus 1)
        lengths: The list of true lengths of the sequences (including eos but
            not bos)
        expected_length_exponent: The exponent to which to raise each segment
            length if expected length regularization is being used. Default: 2.0
        neg_log_inf: The approximation to use for negative (log) infinity.
            Default: -1000000.0
    """
    def __init__(
        self,
        arcs: Tensor,
        lengths: List[int],
        expected_length_exponent: float = 2.0,
        neg_log_inf: float = -1000000.0
    ):
        shape = arcs.size()
        assert len(shape) == 3
        assert len(lengths) == shape[2]

        using_cuda = arcs.is_cuda
        self.device = arcs.get_device if using_cuda else 'cpu'

        self.arcs = arcs
        self.numpy_arcs = self.arcs.clone().detach().cpu().numpy()
        self.lengths = lengths
        self.length_exponent = expected_length_exponent
        self.neg_log_inf = neg_log_inf
        self.max_length = shape[0]
        self.max_seg_len = shape[1]
        self.num_sequences = shape[2]
        self.forward_marginals = None
        self.forward_best = None
        self.backward_marginals = None
        self._best_path = None
        self._best_prob = None
        self.length_index = (
            torch.LongTensor(self.lengths).view(1, -1).to(self.device)
        )

        for seq, length in enumerate(lengths):
            self.arcs[length:, :, seq] = self.neg_log_inf

    def _compute_forward(self):

        # Initialize the forward marginals (aka alphas) to each position in the
        # lattice
        self.forward_marginals = torch.zeros(
            self.max_length + 1,
            self.num_sequences,
            device=self.device,
        )

        # Efficiency hack. We want to be able to add the position-wise marginal
        # to the arcs beginning at that position as a batch, rather than
        # individually. PyTorch complains about gradient calculation if you try
        # to modify the arc matrix in-place, so we separate it into a list of
        # position-wise arc matrices
        forward_arcs = list(self.arcs.clone())

        # Initialize best forward marginal to each position in the lattice, as
        # well as backpointers for computing the best path through the graph
        self.forward_best = np.zeros((self.max_length + 1, self.num_sequences))
        self.backpointers = [{} for x in range(self.num_sequences)]
        for seq in self.backpointers:
            seq[0] = -1

        # Incrementally compute the forward marginals and best marginals for
        # each possible end position in the graph
        for j_end in range(1, self.max_length + 1):
            prefix_probs = []
            best_prefix_probs = []
            previous_candidates = []
            for j_start in range(max(0, j_end - self.max_seg_len), j_end):

                # For each start point that can reach the current j_end, collect
                # the forward marginal to that start point plus the probability
                # of the arc from j_start to j_end
                # prefix_prob = self.forward_marginals[j_start, :]
                # arc_prob = self.arcs[j_start, j_end - j_start - 1, :]
                prefix_probs.append(
                    forward_arcs[j_start][j_end - j_start - 1, :]
                )

                # Additionally collect the best prefix probability to j_start
                # plus the j_start to j_end probability in order to run
                # Viterbi best bath computation
                best_prefix_probs.append(
                    self.numpy_arcs[j_start, j_end - j_start - 1, :]
                )
                previous_candidates.append(j_start)

            # Stack and sum the probabilities of getting to j_end, storing as
            # forward marginal (alpha) of j_end
            prefix_probs = torch.stack(prefix_probs)
            self.forward_marginals[j_end] = torch.logsumexp(
                prefix_probs, dim=0
            ).clamp(min=self.neg_log_inf)
            if j_end < self.max_length:
                forward_arcs[j_end] = (
                    forward_arcs[j_end] + self.forward_marginals[j_end]
                )

            # Stack the best probability candidates to j_end, then find the best
            # and store a backpointer to the node that the best arc came from
            best_prefix_probs = np.stack(best_prefix_probs, axis=0)
            self.forward_best[j_end] = best_prefix_probs.max(axis=0)
            best_previous_length = best_prefix_probs.argmax(axis=0)
            if j_end < self.max_length:
                self.numpy_arcs[j_end] += self.forward_best[j_end]
            best_previous = [
                previous_candidates[k] for k in best_previous_length
            ]
            for i in range(self.num_sequences):
                self.backpointers[i][j_end] = best_previous[i]

    def _compute_backward(self):

        # Initialize the backward marginals (aka betas) to each position in the
        # lattice
        self.backward_marginals = torch.zeros(
            self.max_length + 1, self.num_sequences, device=self.device
        )

        # yapf: disable
        # (It's very difficult to have the style of this section not be
        # terrible)

        # Create "backward padded arcs" for beta computation. Since sequences
        # can be different lengths, the backward marginals need to be "padded"
        # such that they come out to log probability 0 at every position past
        # the actual length of each sequence. Essentially, this means the true
        # probability sum of the outgoing arcs at each of these positions must
        # be 1
        backward_padded_arcs = self.arcs.clone()
        for seq, length in enumerate(self.lengths):
            for k in range(
                1, min(self.max_seg_len, self.max_length - length + 1)
            ):
                backward_padded_arcs[
                    self.max_length - k, :, seq
                ] = np.log(1 / k)
                backward_padded_arcs[
                    self.max_length - k, k:, seq
                ] = self.neg_log_inf
            backward_padded_arcs[
                length:self.max_length - self.max_seg_len + 1, :, seq
            ] = np.log(1 / self.max_seg_len)

        # Shift the arc matrix to represent the *incoming arcs at each position
        # rather than the outgoing ones. This helps us efficiently add up the
        # backward marginal + backward arcs as we go
        expanded_arc_shape = (
            self.max_length + 1, self.max_seg_len, self.num_sequences
        )
        backward_arcs_shifted = (
            torch.empty(expanded_arc_shape).fill_(self.neg_log_inf).to(self.device)
        )
        for seg_len in range(1, self.max_seg_len + 1):
            backward_arcs_shifted[seg_len:, seg_len - 1, :] = (
                backward_padded_arcs[
                    :self.max_length - seg_len + 1, seg_len - 1, :
                ]
            )

        # yapf: enable

        # As in the forward computation, this is mostly a hack. We want to be
        # able to add the position-wise marginal to the arcs ending at that
        # position as a batch, rather than individually. PyTorch complains about
        # gradient calculation if you try to modify the arc matrix in-place, so
        # we separate it into a list of position-wise arc matrices
        backward_padded_arcs = list(backward_arcs_shifted)

        # Incrementally compute the backward marginals for each possible start
        # position in the graph
        for j_start in range(self.max_length - 1, -1, -1):
            suffix_probs = []
            for j_end in range(
                min(self.max_length, j_start + self.max_seg_len), j_start, -1
            ):
                # For each end point that can be reached from  the current
                # j_start, collect the backward marginal to that end point plus
                # the probability of the arc from j_start to j_end
                # suffix_prob = self.backward_marginals[j_end, :]
                # transition_prob = self.arcs[j_start, j_end - j_start - 1, :]
                # suffix_probs.append(suffix_prob + transition_prob)
                suffix_probs.append(
                    backward_padded_arcs[j_end][j_end - j_start - 1, :]
                )

            # Stack and sum the probabilities of transitioning from j_start,
            # storing as the backward marginal (beta) of j_start
            suffix_probs = torch.stack(suffix_probs)
            self.backward_marginals[j_start] = torch.logsumexp(
                suffix_probs, dim=0
            ).clamp(min=self.neg_log_inf)
            backward_padded_arcs[j_start] = (
                backward_padded_arcs[j_start] + self.backward_marginals[j_start]
            )

    def marginal(self) -> Tensor:
        """
        Return the (log) marginal probability of paths through the lattice
        """

        # Computations over the lattice are lazy, so compute the forward
        # pass here if it hasn't been computed already
        if self.forward_marginals is None:
            self._compute_forward()

        # Return the forward marginal as of the ending length of each sequence
        marginals = torch.gather(
            input=self.forward_marginals, dim=0, index=self.length_index
        )
        return marginals

    def best_path(self) -> tuple[list, list]:
        """
        Return the highest-probability path through the lattice
        """

        # Computations over the lattice are lazy, so compute the forward pass
        # here if it hasn't been computed already
        if (self.forward_best is None or self.backpointers is None):
            self._compute_forward()

        # Again, compute the best path and best probability only if they haven't
        # already been computed
        if self._best_path is None or self._best_prob is None:
            self._best_path = []
            self._best_prob = []

            # For each sequence, start at the ending length, cache the final
            # probability at that state, and then follow the backpointers back
            # to the start to find the optimal path
            for i, length in enumerate(self.lengths):
                self._best_prob.append(self.forward_best[length, i])
                position = length
                previous = self.backpointers[i][position]
                path = [(previous, position)]
                while previous > 0:
                    position = previous
                    previous = self.backpointers[i][position]
                    path.append((previous, position))
                path.reverse()
                self._best_path.append(path)

        return self._best_path, self._best_prob

    def expected_length(self) -> Tensor:
        """
        Return the log expected length of any path through the graph
        """

        # Computations over the lattice are lazy, so compute the forward and
        # backward passes here if they have not already been computed
        if self.forward_marginals is None:
            self._compute_forward()
        if self.backward_marginals is None:
            self._compute_backward()
        alpha_beta_probs = []
        length_values = []
        for k in range(1, self.max_seg_len + 1):
            length = k**self.length_exponent
            length_tensor = torch.empty((self.num_sequences)).fill_(length)
            for j_start in range(self.max_length - k + 1):
                length_values.append(length_tensor)
                j_end = j_start + k
                alpha_beta_probs.append(
                    self.forward_marginals[j_start] +
                    self.backward_marginals[j_end] +
                    self.arcs[j_start, k - 1, :]
                )
        alpha_beta_probs = torch.stack(alpha_beta_probs)
        length_values = torch.stack(length_values).to(self.device)
        total_length_expectations = (
            log_expectation(alpha_beta_probs, length_values)
        )
        return total_length_expectations
