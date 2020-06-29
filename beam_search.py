# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of beam search with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7


def log_prob_from_logits(logits, reduce_axis=-1):
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)

def symbols_to_logits(_):
    # Just return random logits
    return tf.random_uniform((batch_size * beam_size, vocab_size))

def beam_search(self):
    batch_size = 2
    beam_size = 3
    vocab_size = 4
    decode_length = 10

    flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])

    flat_logits = symbols_to_logits_fn(flat_ids)

    logits = tf.reshape(flat_logits, [batch_size, beam_size, -1])

    # Convert logits to normalized log probs
    candidate_log_probs = log_prob_from_logits(logits)

    # Multiply the probabilities by the current probabilities of the beam.
    # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
    log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

    length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)

    curr_scores = log_probs / length_penalty
    flat_curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])
    topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2)

    topk_log_probs = topk_scores * length_penalty

    # Work out what beam the top probs are in.
    topk_beam_index = topk_ids // vocab_size
    topk_ids %= vocab_size

    batch_pos = compute_batch_indices(batch_size, beam_size * 2)

    # top beams will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where the
    # last dimension contains the i,j gathering coordinates.
    topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2)

    # Gather up the most probable 2*beams both for the ids and
    # finished_in_alive bools
    topk_seq = tf.gather_nd(alive_seq, topk_coordinates)
    if states:
        states = nest.map_structure(
            lambda state: tf.gather_nd(state, topk_coordinates), states)

    # Append the most probable alive
    topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

    topk_finished = tf.equal(topk_ids, eos_id)

    return topk_seq, topk_log_probs, topk_scores, topk_finished, states
