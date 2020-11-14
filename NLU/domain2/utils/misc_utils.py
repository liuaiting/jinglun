# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import sys
import codecs
import json
import os

import tensorflow as tf


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time. """
    print("%s, time %fs, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def print_out(s, f=None, new_line=True):
    """Similar to print out but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_hparams(hparams, header=None):
    """Print hparams."""
    if header: print_out("%s" % header)
    values = hparams.values()
    for key in sorted(values.keys()):
        print_out("  %s=%s" % (key, str(values[key])))


def load_hparams(model_dir):
    """Load hparams from an existing model directory."""
    hparams_file = os.path.join(model_dir, "hparams")
    if tf.gfile.Exists(hparams_file):
        print_out("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                print_out("  can't load hparams file")
                return None
        return hparams
    else:
        return None


def maybe_parse_standard_hparams(hparams, hparams_path):
    """Override hparams values with existing standard hparams config."""
    if hparams_path and tf.gfile.Exists(hparams_path):
        print_out("# Loading standard hparams from %s" % hparams_path)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_path, "rb")) as f:
            hparams.parse_json(f.read())
    return hparams


def save_hparams(model_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(model_dir, "hparams")
    print_out("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())


def get_config_proto(log_device_placement=False, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True

    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto
