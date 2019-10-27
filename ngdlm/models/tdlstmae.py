from .ae import AE
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from .helpers import append_to_filepath


class TDLSTMAE(AE):
    """ Time-Distributed-LSTM-Autoencoder. This is a autoencoder consisting of an encoder and a decoder. Both are wrapped into TimeDistributed and connected via LSTM. """


    def __init__(
        self,
        encoder=None, decoder=None, autoencoder=None):
        super(TDLSTMAE, self).__init__(encoder=None, decoder=None)

        # Encoder and decoder must be provided.
        assert (encoder != None and decoder != None)

        # From loading.
        if encoder != None and decoder != None and autoencoder != None:
            self.encoder = encoder
            self.decoder = decoder
            self.autoencoder = autoencoder
            return

        # Check preconditions.
        assert len(encoder.outputs) == 1
        assert len(decoder.inputs) == 1
        assert encoder.outputs[0].shape[2:] == decoder.inputs[0].shape[2:] , str(encoder.outputs[0].shape) + " " + str(decoder.inputs[0].shape)
        self.latent_dim = encoder.outputs[0].shape[1]

        # Getting the input-tensors and the LSTM-size.
        encoder_input = encoder.inputs[0]
        decoder_input = decoder.inputs[0]
        lstm_size = decoder_input.shape.as_list()[-1]

        # Time-distributed encoder.
        time_distributed_encoder_input = encoder_input
        time_distributed_encoder_output = layers.TimeDistributed(encoder)(time_distributed_encoder_input)
        time_distributed_encoder = Model(inputs=time_distributed_encoder_input, outputs=time_distributed_encoder_output)
        self.encoder = time_distributed_encoder

        # Time-distributed decoder with LSTM.
        time_distributed_decoder_input = decoder_input
        time_distributed_decoder_output = layers.LSTM(lstm_size, return_sequences=True)(time_distributed_decoder_input)
        time_distributed_decoder_output = layers.TimeDistributed(decoder)(time_distributed_decoder_output)
        time_distributed_decoder = Model(inputs=time_distributed_decoder_input, outputs=time_distributed_decoder_output)
        self.decoder = time_distributed_decoder

        # Autoencoder.
        autoencoder_input = encoder_input
        autoencoder_output = autoencoder_input
        autoencoder_output = time_distributed_encoder(autoencoder_output)
        autoencoder_output = time_distributed_decoder(autoencoder_output)
        self.autoencoder = Model(inputs=autoencoder_input, outputs=autoencoder_output)
