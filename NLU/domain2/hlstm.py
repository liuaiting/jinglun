# -*-coding: utf-8 -*-
"""
Date: 2018-09-25
Author: Liu Aiting
"""
import tensorflow as tf

from NLU.domain2.utils import misc_utils as utils


class HLSTM(object):
    def __init__(self,
                 hparams,
                 iterator,
                 mode,
                 reverse_vocab_table=None,
                 reverse_label_table=None,
                 reverse_intent_table=None):
        self.hparams = hparams
        self.iterator = iterator
        self.mode = mode

        self.reverse_vocab_table = reverse_vocab_table
        self.reverse_label_table = reverse_label_table
        self.reverse_intent_table = reverse_intent_table

        self.embed_size = hparams.embed_size
        self.vocab_size = hparams.vocab_size
        self.max_seq_len = hparams.max_seq_len
        self.num_units = hparams.num_units
        self.label_classes = hparams.label_classes
        self.intent_classes = hparams.intent_classes

        self.max_gradient_norm = hparams.max_gradient_norm
        self.num_keep_ckpts = hparams.num_keep_ckpts

        # Initializer
        initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=hparams.random_seed)
        tf.get_variable_scope().set_initializer(initializer)

        self.init_embedding()
        self.init_inputs()
        self.build_graph()

        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        if self.mode == "train":
            with tf.variable_scope("opt"):
                self.learning_rate = tf.constant(hparams.learning_rate)
                self.learning_rate = self._get_learning_rate_decay(hparams)
                opt = tf.train.AdamOptimizer(self.learning_rate)
                gradients = tf.gradients(self.losses, params)
                clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
                self.grad_norm = grad_norm
                tf.summary.scalar("learning_rate", self.learning_rate)
                tf.summary.scalar("grad_norm", self.grad_norm)
                tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))

        self.merged = tf.summary.merge_all()

        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_keep_ckpts)

        if self.mode != "infer":
            utils.print_out("# Training variables")
            for param in params:
                utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

    def init_embedding(self):
        with tf.variable_scope("embedding"):
            embed_pad = tf.Variable(tf.zeros(shape=[1, self.embed_size]), trainable=False)
            embed_var = tf.get_variable("embed_var", shape=[self.vocab_size - 1, self.embed_size], trainable=True)
            self.embedding = tf.concat([embed_pad, embed_var], axis=0, name="embedding")

    def init_inputs(self):
        with tf.variable_scope("inputs"):
            self.seq = self.iterator.seq
            self.embeded_seq = tf.nn.embedding_lookup(self.embedding, self.seq)
            self.seq_length = self.iterator.seq_length
            self.batch_size = tf.size(self.seq_length)
            if self.mode != "infer":
                self.label = self.iterator.label
                self.intent = self.iterator.intent

    def _get_decay_info(self, hparams):
        """Return decay info based on decay_scheme."""
        if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.5
            if hparams.decay_scheme == "luong5":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 5
            elif hparams.decay_scheme == "luong10":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 10
            elif hparams.decay_scheme == "luong234":
                start_decay_step = int(hparams.num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not hparams.decay_scheme:  # no decay
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif hparams.decay_scheme:
            raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
        return start_decay_step, decay_steps, decay_factor

    def _get_learning_rate_decay(self, hparams):
        """Get learning rate decay."""
        start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
        utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                        "decay_factor %g" % (hparams.decay_scheme,
                                             start_decay_step,
                                             decay_steps,
                                             decay_factor))

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def build_graph(self):
        with tf.variable_scope("hlstm"):
            fw_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units)
            bw_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units)
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=self.embeded_seq,
                sequence_length=self.seq_length,
                dtype=tf.float32)

        with tf.variable_scope("slot"):
            context_rep = tf.concat(bi_outputs, axis=-1)
            W = tf.get_variable("W", shape=[2 * self.num_units, self.label_classes],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.label_classes], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            context_rep_flat = tf.reshape(context_rep, [-1, 2 * self.num_units])  # [batch_size*max_len, hidden_size*2]
            slot_logits = tf.nn.xw_plus_b(context_rep_flat, W, b)
            self.slot_logits = tf.reshape(slot_logits, [self.batch_size, -1, self.label_classes])
            self.slot_pred = tf.to_int32(tf.argmax(self.slot_logits, axis=-1))

        with tf.variable_scope("intent"):
            state_fw = bi_state[0].h
            W = tf.get_variable("W", shape=[self.num_units, self.intent_classes],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.intent_classes], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            intent_rep = tf.reshape(state_fw, [-1, self.num_units])
            intent_logits = tf.nn.xw_plus_b(intent_rep, W, b)
            self.intent_logits = tf.reshape(intent_logits, [-1, self.intent_classes])
            self.intent_pred = tf.to_int32(tf.argmax(self.intent_logits, axis=-1))

        if self.mode == "train":
            with tf.variable_scope("loss"):
                slot_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.slot_logits, labels=self.label)
                mask = tf.sequence_mask(self.seq_length, maxlen=self.max_seq_len, dtype=tf.float32)
                self.slot_loss = tf.reduce_mean(slot_loss * mask)

                intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.intent_logits, labels=self.intent)
                self.intent_loss = tf.reduce_mean(intent_loss)

                self.losses = 0.8 * self.slot_loss + 0.2 * self.intent_loss
                tf.summary.scalar("losses", self.losses)
                tf.summary.scalar("slot_loss", self.slot_loss)
                tf.summary.scalar("intent_loss", self.intent_loss)

        # if self.mode == "eval":
        #     with tf.variable_scope("accuracy"):
        #         self.intent_correct = tf.to_float(tf.equal(self.intent, self.intent_pred))
        #         self.intent_accuracy = tf.reduce_mean(self.intent_correct, name="accuracy")

        with tf.variable_scope("results"):
            if self.mode != "train":
                self.org_seq = self.reverse_vocab_table.lookup(tf.to_int64(self.seq))
                self.pred_label = self.reverse_label_table.lookup(tf.to_int64(self.slot_pred))
                self.pred_intent = self.reverse_intent_table.lookup(tf.to_int64(self.intent_pred))

            if self.mode == "eval":
                self.org_label = self.reverse_label_table.lookup(tf.to_int64(self.label))
                self.org_intent = self.reverse_intent_table.lookup(tf.to_int64(self.intent))

    def train(self, sess):
        return sess.run([
            self.update,
            self.merged,
            self.losses,
            self.slot_loss,
            self.intent_loss,
            self.global_step,
            self.grad_norm,
            self.learning_rate
        ])

    def eval(self, sess):
        return sess.run([
            self.org_seq,
            self.seq_length,
            self.org_label,
            self.org_intent,
            self.pred_label,
            self.pred_intent
        ])

    def infer(self, sess):
        return sess.run([
            self.org_seq,
            self.seq_length,
            self.pred_label,
            self.pred_intent
        ])
