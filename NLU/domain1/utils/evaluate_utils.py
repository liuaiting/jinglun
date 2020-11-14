# -*- coding: utf-8 -*-
"""
Date: 2018-09-26
Author: Liu Aiting
"""
import os
import codecs
import subprocess
import stat

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    """
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

    return compute_label_metrics(filename)


def compute_label_metrics(filename):
    """
    Run conlleval.pl perl script to obtain precision/recall and F1 score.

    Args:
        filename: output file in a specific format.
            https://www.clips.uantwerpen.be/conll2000/chunking/output.html
    Return:
        dict, scores
    """
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                             _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(b''.join(open(filename, "rb").readlines()))
    for line in stdout.split(b'\n'):
        if b'accuracy' in line:
            out = line.split()
            break

    # precision = float(out[6][:-2])
    # recall = float(out[8][:-2])
    accuracy = float(out[1][:-2])
    f1score = float(out[10])
    scores = {"accuracy": accuracy / 100, 'f1': f1score / 100}

    return scores


def compute_intent_metrics(filename):
    df = pd.read_csv(filename, sep="\t", header=None, encoding="utf-8", names=["seq", "intent", "pred"])
    accuracy = accuracy_score(y_true=df["intent"], y_pred=df["pred"])
    f1 = f1_score(y_true=df["intent"], y_pred=df["pred"], average="micro")
    scores = {"accuracy": accuracy, "f1": f1}

    return scores
