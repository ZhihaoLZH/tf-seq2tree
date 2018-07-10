# uncompyle6 version 3.2.3
# Python bytecode 3.5 (3351)
# Decompiled from: Python 3.5.4 (v3.5.4:3f56838, Aug  8 2017, 02:17:05) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: C:\Users\ZH_L\Desktop\ML\seq2tree\seq2tree_model.py
# Compiled at: 2018-06-27 11:38:18
# Size of source mod 2**32: 19969 bytes
import tensorflow as tf, numpy

class Seq2TreeModel:

    def __init__(self,
                mode,
                learning_rate,
                src_vocab_size,
                tgt_vocab_size,
                embedding_size,
                hidden_size,
                sos_id,
                non_terminal_id,
                eos_id,
                left_bracket_id,
                right_bracket_id):

        self.mode = mode

        self.learning_rate = learning_rate

        self.sos_id = sos_id

        self.non_terminal_id = non_terminal_id

        self.eos_id = eos_id

        self.left_bracket_id = left_bracket_id

        self.right_bracket_id = right_bracket_id

        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.src_embeddings = tf.get_variable(name='src_embeddings', shape=[
             src_vocab_size, embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.8, 0.8))

            self.tgt_embeddings = tf.get_variable(name='tgt_embeddings', shape=[
             tgt_vocab_size, embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.8, 0.8))
        self.encoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input_ids')

        self.encoder_input_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='encoder_input_lens')

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            self.encoder = self._create_encoder(self.encoder_input_ids, self.encoder_input_lens, hidden_size)
        encoder_output_states = self.encoder

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.decoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_inputs_ids')
            self.decoder_input_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='decoder_input_lens')
            self.decoder_target_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_target_ids')

        with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
            self.dense = tf.layers.Dense(units=tgt_vocab_size, name='projection_layer', use_bias=False, kernel_initializer=tf.random_uniform_initializer(minval=-0.8, maxval=0.8), bias_initializer=tf.random_uniform_initializer(minval=-0.8, maxval=0.8))

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.decoder = self._create_decoder(self.decoder_input_ids, self.decoder_input_lens, self.decoder_target_ids, encoder_output_states, hidden_size, 2, mode)
            else:
                self.decoder = self._create_decoder(None, None, None, encoder_output_states, hidden_size, 2, mode)

    def _create_single_lstm_cell(self, hidden_size, name, state_is_tuple=False):
        return tf.contrib.rnn.LSTMCell(num_units=hidden_size, name=name, state_is_tuple=state_is_tuple, initializer=tf.random_uniform_initializer(minval=-0.8, maxval=0.8))

    def _create_encoder(self, encoder_input_ids, encoder_input_lens, hidden_size):
        encoder_inputs = tf.nn.embedding_lookup(self.src_embeddings, encoder_input_ids)
        batch_size = tf.shape(encoder_inputs)[0]
        cell_fw = self._create_single_lstm_cell(hidden_size, name='fw_cell')
        cell_bw = self._create_single_lstm_cell(hidden_size, name='bw_cell')
        initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_inputs, sequence_length=encoder_input_lens, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)
        return output_states

    def get_infer_maximum_iterations(self, source_sequence_length):
        decoding_length_factor = 5.0
        max_encoder_length = tf.reduce_max(source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def _create_decoder(self, decoder_input_ids, decoder_input_lens, decoder_target_ids, initial_states, hidden_size, num_layer=2, mode=tf.contrib.learn.ModeKeys.TRAIN):
        cell_list = []
        for _ in range(num_layer):
            c = self._create_single_lstm_cell(hidden_size, name='layer' + str(num_layer) + '_cell')
            cell_list.append(c)

        if num_layer == 1:
            cell = cell_list[0]
        else:
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)

        def train_sample_fn(tgt_in_seq_ids_tgt_in_seq_len_initial_state, max_seq_len, sos_id, non_terminal_id, left_bracket_id):

            def create_zero_array(shape, dtype):
                return tf.zeros(shape=shape, dtype=dtype)

            def cond(
                time,
                max_seq_len,
                unused_tgt_in_seq_ids,
                unused_tgt_in_seq_len,
                unused_predict_ta,
                unused_n_queue_ta,
                unused_q_start_index,
                unused_q_end_index,
                unused_hidden_state,
                unused_sos_id,
                unused_non_terminal_id,
                unused_left_bracket_id):
                return tf.less(time, max_seq_len)

            def run_step(
                    time,
                    max_seq_len,
                    tgt_in_seq_ids,
                    tgt_in_seq_len,
                    predict_ta,
                    n_queue_ta,
                    q_start_index,
                    q_end_index,
                    hidden_state,
                    sos_id,
                    non_terminal_id,
                    left_bracket_id,
            ):
                cur_id = tgt_in_seq_ids[time]
                cur_embed = tf.reshape(tf.nn.embedding_lookup(self.tgt_embeddings, cur_id), shape=[1, -1])

                def true_fn(q_start_index, n_queue_ta, hidden_state):
                    state = n_queue_ta.read(q_start_index)
                    q_start_index = q_start_index + 1
                    return (
                     (
                      state[0][:][:], state[1][:][:]), q_start_index, n_queue_ta)

                def false_fn(q_start_index, n_queue_ta, hidden_state):
                    return (
                     hidden_state, q_start_index, n_queue_ta)

                condition = tf.logical_and(tf.logical_or(tf.equal(cur_id, sos_id), tf.equal(cur_id, left_bracket_id)), tf.less(q_start_index, q_end_index))
                pre_state, q_start_index, n_queue_ta = tf.cond(condition, true_fn=lambda : true_fn(q_start_index, n_queue_ta, hidden_state), false_fn=lambda : false_fn(q_start_index, n_queue_ta, hidden_state))
                call_cell = lambda : cell.call(cur_embed, pre_state)

                def output_state_false_fn(pre_state):
                    return (
                     create_zero_array(shape=[1, cell.output_size], dtype=tf.float32), pre_state)

                new_output, new_state = tf.cond(tf.less(time, tgt_in_seq_len), true_fn=call_cell, false_fn=lambda : output_state_false_fn(pre_state))
                new_output = tf.reshape(new_output, shape=[cell.output_size])
                predict_ta = predict_ta.write(time, new_output)

                def true_fn1(state, n_queue_ta, q_end_index):
                    n_queue_ta = n_queue_ta.write(q_end_index, state)
                    q_end_index = q_end_index + 1
                    return (
                     q_end_index, n_queue_ta)

                def false_fn1(q_end_index, n_queue_ta):
                    return (
                     q_end_index, n_queue_ta)

                q_end_index, n_queue_ta = tf.cond(tf.equal(cur_id, non_terminal_id), true_fn=lambda : true_fn1(new_state, n_queue_ta, q_end_index), false_fn=lambda : false_fn1(q_end_index, n_queue_ta))
                return (
                 time + 1, max_seq_len, tgt_in_seq_ids, tgt_in_seq_len, predict_ta, n_queue_ta, q_start_index, q_end_index, new_state, sos_id, non_terminal_id)

            tgt_in_seq_ids, tgt_in_seq_len, initial_state = tgt_in_seq_ids_tgt_in_seq_len_initial_state
            initial_state = (
             tf.reshape(initial_state[0], [1, -1]), tf.reshape(initial_state[1], [1, -1]))
            predict_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            n_queue_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            q_start_index = tf.constant(0, dtype=tf.int32)
            q_end_index = tf.constant(0, dtype=tf.int32)
            time = tf.constant(0, dtype=tf.int32)
            n_queue_ta = n_queue_ta.write(q_end_index, initial_state)
            q_end_index = q_end_index + 1
            _, _, _, _, predict_ta, _, _, _, _, _, _, _ = tf.while_loop(cond=cond, body=run_step, loop_vars=[
             time,
             max_seq_len,
             tgt_in_seq_ids,
             tgt_in_seq_len,
             predict_ta,
             n_queue_ta,
             q_start_index,
             q_end_index,
             initial_state,
             sos_id,
             non_terminal_id,
             left_bracket_id])
            stacked_predict = predict_ta.stack()
            return stacked_predict

        dense = self.dense

        def infer(initial_state, maximum_iterations, sos_id, non_terminal_id, eos_id, left_bracket_id, right_bracket_id):

            def cond2(
                    time,
                    maximum_iterations,
                    unused_pre_id,
                    unused_predict_ta,
                    unused_n_queue_ta,
                    unused_q_start_index,
                    unused_q_end_index,
                    unused_hidden_state,
                    unused_sos_id,
                    unused_non_terminal_id,
                    unused_eos_id,
                    unused_left_bracket_id,
                    unused_right_bracket_id
            ):
                return tf.less(time, maximum_iterations)

            def run_step2(
                    time,
                    maximum_iterations,
                    pre_id,
                    predict_ta,
                    n_queue_ta,
                    q_start_index,
                    q_end_index,
                    hidden_state,
                    sos_id,
                    non_terminal_id,
                    eos_id,
                    left_bracket_id,
                    right_bracket_id,
            ):
                cur_id = pre_id
                cur_embed = tf.reshape(tf.nn.embedding_lookup(self.tgt_embeddings, cur_id), shape=[1, -1])

                def infer_true_fn(q_start_index, n_queue_ta, hidden_state):
                    state = n_queue_ta.read(q_start_index)
                    q_start_index = q_start_index + 1
                    return (
                     (
                      state[0][:][:], state[1][:][:]), q_start_index, n_queue_ta)

                def infer_false_fn(q_start_index, n_queue_ta, hidden_state):
                    return (
                     hidden_state, q_start_index, n_queue_ta)

                condition = tf.logical_and(tf.logical_or(tf.equal(cur_id, sos_id), tf.equal(cur_id, left_bracket_id)), tf.less(q_start_index, q_end_index))
                pre_state, q_start_index, n_queue_ta = tf.cond(condition, true_fn=lambda : infer_true_fn(q_start_index, n_queue_ta, hidden_state), false_fn=lambda : infer_false_fn(q_start_index, n_queue_ta, hidden_state))
                call_cell = lambda : cell.call(cur_embed, pre_state)
                new_output, new_state = call_cell()
                print('new_output:', new_output)
                logit = dense(new_output)
                print('logit:', logit)
                output_id = tf.reshape(tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32), shape=())
                print('output_id:', output_id)

                def infer_true_fn1(output_id, left_bracket_id):
                    return left_bracket_id

                def infer_false_fn1(output_id, left_bracket_id):
                    return output_id

                output_id = tf.cond(tf.equal(output_id, eos_id), true_fn=lambda : infer_true_fn1(output_id, left_bracket_id), false_fn=lambda : infer_false_fn1(output_id, left_bracket_id))
                logit = tf.reshape(logit, shape=[tf.shape(logit)[-1]])
                predict_ta = predict_ta.write(time, logit)

                def infer_true_fn2(state, n_queue_ta, q_end_index):
                    n_queue_ta = n_queue_ta.write(q_end_index, state)
                    q_end_index = q_end_index + 1
                    return (
                     q_end_index, n_queue_ta)

                def infer_false_fn2(q_end_index, n_queue_ta):
                    return (
                     q_end_index, n_queue_ta)

                q_end_index, n_queue_ta = tf.cond(tf.equal(cur_id, non_terminal_id), true_fn=lambda : infer_true_fn2(new_state, n_queue_ta, q_end_index), false_fn=lambda : infer_false_fn2(q_end_index, n_queue_ta))
                return (
                 time + 1, maximum_iterations, output_id, predict_ta, n_queue_ta, q_start_index, q_end_index, new_state, sos_id, non_terminal_id, eos_id, right_bracket_id)

            initial_state = (
             tf.reshape(initial_state[0], [1, -1]), tf.reshape(initial_state[1], [1, -1]))
            predict_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            n_queue_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            q_start_index = tf.constant(0, dtype=tf.int32)
            q_end_index = tf.constant(0, dtype=tf.int32)
            time = tf.constant(0, dtype=tf.int32)
            n_queue_ta = n_queue_ta.write(q_end_index, initial_state)
            q_end_index = q_end_index + 1
            _, _, _, predict_ta, _, _, _, _, _, _, _, _, _ = tf.while_loop(cond=cond2, body=run_step2, loop_vars=[
             time,
             maximum_iterations,
             sos_id,
             predict_ta,
             n_queue_ta,
             q_start_index,
             q_end_index,
             initial_state,
             sos_id,
             non_terminal_id,
             eos_id,
             left_bracket_id,
             right_bracket_id])
            stacked_predict = predict_ta.stack()
            return stacked_predict

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            shape = tf.shape(decoder_input_ids)
            batch_size = shape[0]
            max_seq_len = shape[1]
            p = tf.map_fn(fn=lambda x: train_sample_fn(x, max_seq_len, self.sos_id, self.non_terminal_id, self.left_bracket_id), elems=(decoder_input_ids, decoder_input_lens, initial_states), dtype=tf.float32)
            print('p', p)
            logits = self.dense(p)
            print('logits:', logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=decoder_target_ids)
            tgt_weights = tf.sequence_mask(self.decoder_input_lens, max_seq_len, dtype=logits.dtype)
            loss = tf.reduce_sum(losses * tgt_weights) / tf.to_float(batch_size)
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            self.loss = loss
            clip_grads, grad_norm = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).apply_gradients(zip(clip_grads, params))
        else:
            maximum_iterations = self.get_infer_maximum_iterations(self.encoder_input_lens)
            p = tf.map_fn(fn=lambda x: infer(x, maximum_iterations, self.sos_id, self.non_terminal_id, self.eos_id, self.left_bracket_id, self.right_bracket_id), elems=initial_states, dtype=tf.float32)
            logits = p
            print('logits:', logits)
            self.predict_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            print('predict_ids:', self.predict_ids)

    def train(self, sess, src_batch, tgt_batch):
        if not self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            raise AssertionError
        batch_src_seqs, batch_src_lens = src_batch
        batch_tgt_in_seqs, batch_tgt_out_seqs, batch_tgt_in_lens = tgt_batch
        feed_dict = {self.encoder_input_ids: numpy.array(batch_src_seqs),
         self.encoder_input_lens: numpy.array(batch_src_lens),
         self.decoder_input_ids: numpy.array(batch_tgt_in_seqs),
         self.decoder_input_lens: numpy.array(batch_tgt_in_lens),
         self.decoder_target_ids: numpy.array(batch_tgt_out_seqs)}
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def decode(self, sess, src_batch):
        if not self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            raise AssertionError
        batch_src_seqs, batch_src_lens = src_batch

        feed_dict = {self.encoder_input_ids: numpy.array(batch_src_seqs),
         self.encoder_input_lens: numpy.array(batch_src_lens)}

        predict_ids = sess.run(self.predict_ids, feed_dict=feed_dict)
        return predict_ids
# okay decompiling C:/Users/ZH_L/Desktop/ML/seq2tree\seq2tree_model.cpython-35.pyc
