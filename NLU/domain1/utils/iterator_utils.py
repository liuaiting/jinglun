# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from . import vocab_utils

COLUMNS_TRAIN = ["seq", "label", "intent"]
# COLUMNS_TEST = ["seq"]

FIELD_DEFAULT_TRAIN = [[""], [""], [""]]
# FIELD_DEFAULT_TEST = [[""]]


def _parse_line_train(line):
    fields = tf.decode_csv(line, FIELD_DEFAULT_TRAIN, field_delim="\t")
    columns = dict(zip(COLUMNS_TRAIN, fields))
    seq = columns.pop("seq")
    label = columns.pop("label")
    intent = columns.pop("intent")
    return seq, label, intent


# def _parse_line_test(line):
#     fields = tf.decode_csv(line, FIELD_DEFAULT_TEST, field_delim="\t")
#     columns = dict(zip(COLUMNS_TEST, fields))
#     seq = columns.pop("seq")
#     return seq


class BatchedInput(
    collections.namedtuple("BatchTrainInput",
                           ("initializer",
                            "label", "intent",
                            "seq", "seq_length"))):
    pass


def get_iterator(dataset,
                 vocab_table,
                 label_table,
                 intent_table,
                 batch_size,
                 random_seed=None,
                 num_buckets=1,
                 max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 reshuffle_each_iteration=True):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    dataset = dataset.map(_parse_line_train)

    dataset = dataset.shuffle(
        output_buffer_size, random_seed, reshuffle_each_iteration)

    dataset = dataset.map(
        lambda seq, label, intent: (
            tf.string_split([seq]).values, tf.string_split([label]).values, intent),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    dataset = dataset.filter(
        lambda seq, label, intent: tf.size(seq) > 0)

    if max_len:
        dataset = dataset.map(
            lambda seq, label, intent: (seq[:max_len], label[:max_len], intent),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids. Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(
        lambda seq, label, intent: (tf.cast(vocab_table.lookup(seq), tf.int32),
                                    tf.cast(label_table.lookup(label), tf.int32),
                                    tf.cast(intent_table.lookup(intent), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    dataset = dataset.map(
        lambda seq, label, intent: (seq, label, intent, tf.size(seq)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Bucket by seq sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([max_len]),
                tf.TensorShape([max_len]),
                tf.TensorShape([]),
                tf.TensorShape([])),
            padding_values=(
                vocab_utils.PAD_ID,
                0,  # O_id
                0,  # unused
                0)  # unused
        )

    if num_buckets > 1:
        def key_func(unused_1, unuesd_2, unused_3, seq_lens):
            if max_len:
                bucket_width = (max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            bucket_id = seq_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(dataset)

    batch_iter = batched_dataset.make_initializable_iterator()
    (seq_ids, label, intent, seq_len) = batch_iter.get_next()

    return BatchedInput(
        initializer=batch_iter.initializer,
        seq=seq_ids,
        label=label,
        intent=intent,
        seq_length=seq_len)


def get_infer_iterator(dataset,
                       vocab_table,
                       batch_size,
                       max_len=None):

    dataset = dataset.map(
        lambda seq: tf.string_split([seq]).values)

    if max_len:
        dataset = dataset.map(
            lambda seq: seq[:max_len])

    dataset = dataset.map(
        lambda seq: (
            tf.cast(vocab_table.lookup(seq), tf.int32)))

    dataset = dataset.map(
        lambda seq: (
            seq, tf.size(seq)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([max_len]),
                tf.TensorShape([])),
            padding_values=(
                vocab_utils.PAD_ID,
                0))

    batched_dataset = batching_func(dataset)
    batch_iter = batched_dataset.make_initializable_iterator()
    (seq_ids, seq_len) = batch_iter.get_next()

    return BatchedInput(
        initializer=batch_iter.initializer,
        seq=seq_ids,
        seq_length=seq_len,
        label=None,
        intent=None
    )


def get_eval_iterator(dataset,
                      vocab_table,
                      label_table,
                      intent_table,
                      batch_size,
                      max_len=None):

    dataset = dataset.map(_parse_line_train)

    dataset = dataset.map(
        lambda seq, label, intent: (
            tf.string_split([seq]).values, tf.string_split([label]).values, intent))

    # Filter zero length input sequences.
    dataset = dataset.filter(
        lambda seq, label, intent: tf.size(seq) > 0)

    if max_len:
        dataset = dataset.map(
            lambda seq, label, intent: (seq[:max_len], label[:max_len], intent))

    # Convert the word strings to ids. Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(
        lambda seq, label, intent: (tf.cast(vocab_table.lookup(seq), tf.int32),
                                    tf.cast(label_table.lookup(label), tf.int32),
                                    tf.cast(intent_table.lookup(intent), tf.int32)))

    # Add in sequence lengths.
    dataset = dataset.map(
        lambda seq, label, intent: (seq, label, intent, tf.size(seq)))

    # Bucket by seq sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([max_len]),
                tf.TensorShape([max_len]),
                tf.TensorShape([]),
                tf.TensorShape([])),
            padding_values=(
                vocab_utils.PAD_ID,
                0,  # O_id
                0,  # unused
                0)  # unused
        )

    batched_dataset = batching_func(dataset)

    batch_iter = batched_dataset.make_one_shot_iterator()
    (seq_ids, label, intent, seq_len) = batch_iter.get_next()

    return BatchedInput(
        initializer=batch_iter.initializer,
        seq=seq_ids,
        label=label,
        intent=intent,
        seq_length=seq_len)
