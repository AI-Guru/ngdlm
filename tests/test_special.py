import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers
import logging
import numpy as np
import sys

class TestSpecial(unittest.TestCase):

    def test_time_distributed_ae(self):

        # Encoder.
        encoder_input = layers.Input((None, 1024))
        encoder_output = encoder_input
        encoder_output = layers.Dense(128)(encoder_output)
        encoder_output = layers.Dense(64)(encoder_output)
        encoder = models.Model(inputs=encoder_input, outputs=encoder_output)

        # Decoder.
        decoder_input = layers.Input((None, 64))
        decoder_output = decoder_input
        decoder_output = layers.Dense(128)(decoder_output)
        decoder_output = layers.Dense(1024)(decoder_output)
        decoder = models.Model(inputs=decoder_input, outputs=decoder_output)

        # Autoencoder.
        tdlstmae = ngdlmodels.TDLSTMAE(encoder=encoder, decoder=decoder)
        tdlstmae.compile(loss="mse", optimizer="adam")
        tdlstmae.summary()

        # Train model.
        x_train = np.ones((10, 100, 1024))
        y_train = np.zeros((10, 100, 1024))
        history = tdlstmae.fit(
            x_train, y_train,
            epochs=1,
            batch_size=16
            )


    def test_cae(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(2, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Autoencoder.
        cae = ngdlmodels.CAE(encoder=encoder, decoder=decoder)
        cae.compile(loss="mse", optimizer="adam")
        cae.summary()

        # Train model.
        x_train = np.ones((10, 100))
        y_train = np.zeros((10, 100))
        history = cae.fit(
            x_train, y_train,
            epochs=1,
            batch_size=16
            )


if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    unittest.main()
