# -*- coding: utf-8 -*-
"""
Date: 2018-09-06
Author: aiting
"""
from __future__ import division
from __future__ import print_function

import collections
import time

import tensorflow as tf
import numpy as np

from . import vocab_utils
from . import iterator_utils
from . import misc_utils as utils


def print_variables_in_ckpt(ckpt_path):
    """Print a list of variables in a checkpoint together with their shapes."""
    utils.print_out("# Variables in ckpt %s" % ckpt_path)
    reader = tf.train.NewCheckpointReader(ckpt_path)
    variable_map = reader.get_variable_to_shape_map()
    for key in sorted(variable_map.keys()):
        utils.print_out("  %s: %s" % (key, variable_map[key]))


def load_model(model, ckpt_path, session, name):
    """Load model from a checkpoint."""
    start_time = time.time()
    try:
        model.saver.restore(session, ckpt_path)
    except tf.errors.NotFoundError as e:
        utils.print_out("Can't load checkpoint")
        print_variables_in_ckpt(ckpt_path)
        utils.print_out("%s" % str(e))

    session.run(tf.tables_initializer())
    # utils.print_out(
    #     "# loaded %s model parameters from %s, time %.2fs" %
    #     (name, ckpt_path, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                        (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


class Model(collections.namedtuple("Model", ("graph", "model", "iterator"))):
    pass


class InferModel(collections.namedtuple("Model", ("graph", "model", "iterator",
                                                  "data_placeholder"))):
    pass


def create_train_model(model_creator, hparams, mode="train"):
    vocab_file = hparams.vocab_file
    label_file = hparams.label_file
    intent_file = hparams.intent_file

    assert mode == "train"
    data_file = hparams.train_file
    iterator_creator = iterator_utils.get_iterator

    graph = tf.Graph()

    with graph.as_default(), tf.container(mode):
        with tf.name_scope("vocab_table"):
            vocab_table = vocab_utils.create_vocab_table(vocab_file)
            label_table = vocab_utils.create_label_table(label_file)
            intent_table = vocab_utils.create_label_table(intent_file)

        with tf.name_scope("iterator"):
            dataset = tf.data.TextLineDataset(data_file)

            iterator = iterator_creator(
                dataset,
                vocab_table,
                label_table,
                intent_table,
                hparams.batch_size,
                random_seed=hparams.random_seed,
                num_buckets=hparams.num_buckets,
                max_len=hparams.max_seq_len)

        # Model
        model = model_creator(hparams,
                              iterator,
                              mode=mode)
        return Model(
            graph=graph,
            model=model,
            iterator=iterator)


def create_eval_model(model_creator, hparams, mode="eval"):
    vocab_file = hparams.vocab_file
    label_file = hparams.label_file
    intent_file = hparams.intent_file

    assert mode == "eval"
    data_file = hparams.dev_file
    iterator_creator = iterator_utils.get_eval_iterator

    graph = tf.Graph()

    with graph.as_default(), tf.container(mode):
        with tf.name_scope("vocab_table"):
            vocab_table = vocab_utils.create_vocab_table(vocab_file)
            label_table = vocab_utils.create_label_table(label_file)
            intent_table = vocab_utils.create_label_table(intent_file)
            reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
                vocab_file, default_value=vocab_utils.UNK)
            reverse_label_table = tf.contrib.lookup.index_to_string_table_from_file(
                label_file, default_value=vocab_utils.UNK)
            reverse_intent_table = tf.contrib.lookup.index_to_string_table_from_file(
                intent_file, default_value=vocab_utils.UNK)

        with tf.name_scope("iterator"):
            dataset = tf.data.TextLineDataset(data_file)

            iterator = iterator_creator(
                dataset,
                vocab_table,
                label_table,
                intent_table,
                hparams.batch_size)

        # Model
        model = model_creator(hparams,
                              iterator,
                              mode=mode,
                              reverse_vocab_table=reverse_vocab_table,
                              reverse_label_table=reverse_label_table,
                              reverse_intent_table=reverse_intent_table)
        return Model(
            graph=graph,
            model=model,
            iterator=iterator)


def create_infer_model(model_creator, hparams, mode="infer"):
    vocab_file = hparams.vocab_file
    label_file = hparams.label_file
    intent_file = hparams.intent_file

    assert mode == "infer"
    # data_file = hparams.test_file
    iterator_creator = iterator_utils.get_infer_iterator

    graph = tf.Graph()

    with graph.as_default(), tf.container(mode):
        with tf.name_scope("vocab_table"):
            vocab_table = vocab_utils.create_vocab_table(vocab_file)
            reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
                vocab_file, default_value=vocab_utils.UNK)
            reverse_label_table = tf.contrib.lookup.index_to_string_table_from_file(
                label_file, default_value=vocab_utils.UNK)
            reverse_intent_table = tf.contrib.lookup.index_to_string_table_from_file(
                intent_file, default_value=vocab_utils.UNK)

        with tf.name_scope("iterator"):
            data_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)

            iterator = iterator_creator(
                dataset,
                vocab_table,
                hparams.infer_batch_size)

        # Model
        model = model_creator(hparams,
                              iterator,
                              mode=mode,
                              reverse_vocab_table=reverse_vocab_table,
                              reverse_label_table=reverse_label_table,
                              reverse_intent_table=reverse_intent_table)
        return InferModel(
            graph=graph,
            model=model,
            iterator=iterator,
            data_placeholder=data_placeholder)
