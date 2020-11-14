# -*- coding: utf-8 -*-
"""Utility to handle vocabulary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import time

import tensorflow as tf

from . import misc_utils as utils

PAD = "<pad>"
UNK = "<unk>"
PAD_ID = 0
UNK_ID = 1


def naive_tokenizer(line):
    return line.strip().lower().split()


def create_vocab(in_path, out_path, max_size=None, min_freq=1, tokenizer=None):
    if not tf.gfile.Exists(out_path):
        start_time = time.time()
        utils.print_out("Creating vocabulary {} from data {}".format(out_path, in_path))
        vocab = collections.Counter()
        with open(in_path, mode='r') as f:
            for line in f:
                line = line.strip().split("\t")[0]
                tokens = tokenizer(line) if tokenizer else naive_tokenizer(line)
                vocab.update(tokens)
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
            sorted_vocab.sort(key=lambda x: x[1], reverse=True)
            itos = [PAD, UNK]
            for word, freq in sorted_vocab:
                if freq < min_freq or len(itos) == max_size:
                    break
                itos.append(word)
            with open(out_path, mode='w') as fw:
                for word in itos:
                    fw.write(str(word) + '\n')

        utils.print_out("  PAD word id is %s." % PAD_ID)
        utils.print_out("  Unknown word id is %s." % UNK_ID)
        utils.print_out("  size of vocabulary is %s. " % len(itos))
        utils.print_out("  min frequency is %d. " % min_freq)

        utils.print_time("  create vocab ", start_time)
    else:
        utils.print_out("Vocab file %s already exists." % out_path)


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


def create_vocab_table(text_vocab_file):
    """Create vocab table for text_file."""
    text_vocab_table = tf.contrib.lookup.index_table_from_file(
        text_vocab_file, default_value=UNK_ID)
    return text_vocab_table


def create_label_table(text_vocab_file):
    """Create vocab table for text_file."""
    text_vocab_table = tf.contrib.lookup.index_table_from_file(
        text_vocab_file, default_value=0)
    return text_vocab_table
