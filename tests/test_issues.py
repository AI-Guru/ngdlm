import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers
import logging
import numpy as np
import glob
import os


class TestModels(unittest.TestCase):

    def test_issue_1(self):
        input_shape = (28, 28)
        latent_dim = 128

        encoder_input = layers.Input(shape=input_shape)
        encoder_output = encoder_input
        encoder_output = layers.Reshape(input_shape + (1,))(encoder_input)
        encoder_output = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoder_output)
        encoder_output = layers.MaxPooling2D((2, 2), padding="same")(encoder_output)
        encoder_output = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoder_output)
        encoder_output = layers.MaxPooling2D((2, 2), padding="same")(encoder_output)
        encoder_output = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoder_output)
        encoder_output = layers.MaxPooling2D((2, 2), padding="same")(encoder_output)
        encoder_output = layers.Flatten()(encoder_output)
        encoder = models.Model(encoder_input, encoder_output)

        # Create the decoder.
        decoder_input = layers.Input(shape=(latent_dim,))
        decoder_output = decoder_input
        #decoder_output = layers.Dense(128, activation="relu")(decoder_output)
        decoder_output = layers.Reshape((4, 4, 8))(decoder_output)
        decoder_output = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(decoder_output)
        decoder_output = layers.UpSampling2D((2, 2))(decoder_output)
        decoder_output = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(decoder_output)
        decoder_output = layers.UpSampling2D((2, 2))(decoder_output)
        decoder_output = layers.Conv2D(16, (3, 3), activation="relu")(decoder_output)
        decoder_output = layers.UpSampling2D((2, 2))(decoder_output)
        decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(decoder_output)
        decoder_output = layers.Reshape((28, 28))(decoder_output)
        decoder = models.Model(decoder_input, decoder_output)

        # Create the VAE.
        vae = ngdlmodels.VAE(encoder, decoder, latent_dim=latent_dim)
        vae.compile(optimizer='adadelta', loss="binary_crossentropy")
        vae.summary()

        # Train.
        print("Train...")
        history = vae.fit(
                x_input_train, x_input_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(x_input_test, x_input_test)
            )

        # Evaluate.
        #print("Evaluate...")
        #loss = vae.model.evaluate(x_input_test, x_input_test)
        #print("Loss:", loss)


if __name__ == '__main__':
    unittest.main()
