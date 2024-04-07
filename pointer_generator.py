import tensorflow as tf
from tensorflow.python.keras import backend as K

logger = tf.get_logger()

class AttentionLayer(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, tuple)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == tuple
        encoder_out_seq, decoder_out_seq = inputs

        logger.debug(f"encoder_out_seq.shape = {encoder_out_seq.shape}")
        logger.debug(f"decoder_out_seq.shape = {decoder_out_seq.shape}")

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            logger.debug("Running energy computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_full_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            logger.debug(f"U_a_dot_h.shape = {U_a_dot_h.shape}")

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)

            logger.debug(f"Ws_plus_Uh.shape = {Ws_plus_Uh.shape}")

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            logger.debug(f"ei.shape = {e_i.shape}")

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            logger.debug("Running attention vector computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_full_seq * K.expand_dims(inputs, -1), axis=1)

            logger.debug(f"ci.shape = {c_i.shape}")

            return c_i, [c_i]

        # we don't maintain states between steps when computing attention
        # attention is stateless, so we're passing a fake state for RNN step function
        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim
        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e], constants=[encoder_out_seq]
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c], constants=[encoder_out_seq]
        )
#         print(c_outputs.shape)
        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

    
class PointerGeneratorLayer(tf.keras.layers.Layer):
    def __init__(self, title_vocab_size, simple_convert, title_len, abs_len, **kwargs):
        super(PointerGeneratorLayer, self).__init__(**kwargs)
        self.title_vocab_size = title_vocab_size
        self.title_len = title_len
        self.abs_len = abs_len
        self.simple_convert = simple_convert  # TensorFlow hashmap for token conversion

    def build(self, input_shape):
        assert isinstance(input_shape, tuple)
        # Create a trainable weight variable for this layer.
        super(PointerGeneratorLayer, self).build(input_shape)  # Be sure to call this at the end  
    
    def call(self, inputs):
        decoder_outputs, attention_scores, input_sequence, repeat_idx, repeat_idx2 = inputs
        batch_size = tf.shape(attention_scores)[0]
        sequence_len= tf.shape(attention_scores)[1]
        
        
        attention_scores = tf.reshape(attention_scores, [-1])
        
        # convert the input sequence and apply it to each word of the input sequence (like the decoder dist is)
        input_sequence = tf.cast(input_sequence, tf.int32)
        input_sequence = self.simple_convert.lookup(input_sequence)
        input_sequence = tf.repeat(input_sequence, repeats = sequence_len, axis = 0)
        input_sequence = tf.reshape(input_sequence, [tf.shape(input_sequence)[0],tf.shape(input_sequence)[1], 1])
        
        # add the title indicies (basically so that the update later knows exactly which word in the title/input sequence to update to)
        repeat_index = repeat_idx[:sequence_len]
        tiled_index = tf.tile(repeat_index, [(tf.shape(input_sequence)[0]*tf.shape(input_sequence)[1]/tf.shape(repeat_index)[0]), 1])
        tiled_index = tf.reshape(tiled_index, [tf.shape(input_sequence)[0],tf.shape(input_sequence)[1], 1])
        input_sequence = tf.concat([tiled_index, input_sequence], axis=2)
        
        # add the batch number (so the update later knows which batch to update to)
        repeat_index = repeat_idx2[:batch_size]
        tiled_index = tf.repeat(repeat_index, repeats = tf.shape(input_sequence)[1]*sequence_len, axis=0) 
        tiled_index = tf.reshape(tiled_index, [tf.shape(input_sequence)[0],tf.shape(input_sequence)[1], 1])
        input_sequence = tf.concat([tiled_index, input_sequence], axis=2)
        input_sequence = tf.reshape(input_sequence, [batch_size*self.abs_len*sequence_len,3])
        
        
        best_dist = tf.tensor_scatter_nd_max(decoder_outputs, input_sequence, attention_scores)
        
        kill_mask = tf.ones(batch_size*sequence_len, tf.int32)
        kill_values = tf.zeros(batch_size*sequence_len, tf.float32)

        kill_mask = tf.reshape(kill_mask, [batch_size*sequence_len, 1])
        repeat_index = repeat_idx[:sequence_len]
        tiled_index = tf.tile(repeat_index, [(tf.shape(kill_mask)[0]/sequence_len), 1])
        kill_mask = tf.concat([tiled_index, kill_mask], axis=1)

        repeat_index = repeat_idx2[:batch_size]
        tiled_index = tf.repeat(repeat_index, repeats = sequence_len, axis=0)
        kill_mask = tf.concat([tiled_index, kill_mask], axis=1)

        best_dist = tf.tensor_scatter_nd_update(best_dist, kill_mask, kill_values)
                              
        return best_dist
    
# works
#     def call(self, inputs):
#         decoder_outputs, attention_scores, input_sequence, repeat_idx, repeat_idx2 = inputs
#         batch_size = tf.shape(attention_scores)[0]
#         sequence_len= tf.shape(attention_scores)[1]
        
        
#         attention_scores = tf.reshape(attention_scores, [-1])
        
#         # convert the input sequence and apply it to each word of the input sequence (like the decoder dist is)
#         input_sequence = tf.cast(input_sequence, tf.int32)
#         input_sequence = self.simple_convert.lookup(input_sequence)
#         input_sequence = tf.repeat(input_sequence, repeats = sequence_len, axis = 0)
#         input_sequence = tf.reshape(input_sequence, [tf.shape(input_sequence)[0],tf.shape(input_sequence)[1], 1])
        
#         # add the title indicies (basically so that the update later knows exactly which word in the title/input sequence to update to)
#         repeat_index = repeat_idx[:sequence_len]
#         tiled_index = tf.tile(repeat_index, [(tf.shape(input_sequence)[0]*tf.shape(input_sequence)[1]/tf.shape(repeat_index)[0]), 1])
#         tiled_index = tf.reshape(tiled_index, [tf.shape(input_sequence)[0],tf.shape(input_sequence)[1], 1])
#         input_sequence = tf.concat([tiled_index, input_sequence], axis=2)
        
#         # add the batch number (so the update later knows which batch to update to)
#         repeat_index = repeat_idx2[:batch_size]
#         tiled_index = tf.repeat(repeat_index, repeats = tf.shape(input_sequence)[1]*sequence_len, axis=0) 
#         tiled_index = tf.reshape(tiled_index, [tf.shape(input_sequence)[0],tf.shape(input_sequence)[1], 1])
#         input_sequence = tf.concat([tiled_index, input_sequence], axis=2)
#         input_sequence = tf.reshape(input_sequence, [batch_size*self.abs_len*sequence_len,3])
        
        
#         best_dist = tf.tensor_scatter_nd_max(decoder_outputs, input_sequence, attention_scores)
        
#         return best_dist
    
    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return tf.TensorShape((input_shape[0][0], input_shape[0][1], input_shape[0][2]))
        