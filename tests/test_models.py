import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers

class TestModels(unittest.TestCase):


    def test_ae_functional(self):

        # Encoder.
        encoder_input = layers.Input(shape=(100,))
        encoder_output = encoder_input
        encoder_output = layers.Dense(2, activation="relu")(encoder_output)

        # Decoder.
        decoder_input = layers.Input(shape=(2,))
        decoder_output = layers.Dense(100, activation="sigmoid")(decoder_input)

        # Autoencoder.
        ae = ngdlmodels.AE(
            encoder_input=encoder_input, encoder_output=encoder_output,
            decoder_input=decoder_input, decoder_output=decoder_output)
        ae.summary()


    def test_ae_sequential(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(2, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Autoencoder.
        ae = ngdlmodels.AE(encoder=encoder, decoder=decoder)
        ae.summary()


    def test_vae_functional(self):

        # Encoder.
        encoder_input = layers.Input(shape=(100,))
        encoder_output = encoder_input
        encoder_output = layers.Dense(20, activation="relu")(encoder_output)

        # Decoder.
        decoder_input = layers.Input(shape=(2,))
        decoder_output = layers.Dense(20, activation="relu")(decoder_input)
        decoder_output = layers.Dense(100, activation="sigmoid")(decoder_output)

        # Variational Autoencoder.
        vae = ngdlmodels.VAE(
            encoder_input=encoder_input, encoder_output=encoder_output,
            decoder_input=decoder_input, decoder_output=decoder_output,
            latent_dim=2)
        vae.summary()


    def test_vae_sequential(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Variational Autoencoder.
        vae = ngdlmodels.VAE(encoder=encoder, decoder=decoder, latent_dim=2)
        vae.summary()


    def test_tl_functional(self):

        # Base model.
        base_model_input = layers.Input(shape=(100,))
        base_model_output = base_model_input
        base_model_output = layers.Dense(20, activation="relu")(base_model_output)

        # Decoder.
        decoder_input = layers.Input(shape=(2,))
        decoder_output = layers.Dense(20, activation="relu")(decoder_input)
        decoder_output = layers.Dense(100, activation="sigmoid")(decoder_output)

        # Triplet loss.
        tl = ngdlmodels.TL(base_input=base_model_input, base_output=base_model_output)
        tl.summary()


    def test_tl_sequential(self):

        # Base model.
        base_model = models.Sequential()
        base_model.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Triplet loss.
        tl = ngdlmodels.TL(base=base_model)
        tl.summary()

    #def test_ae_save_load(self):

    #    # Encoder.
    #    encoder_input = layers.Input(shape=(100,))
    #    encoder_output = encoder_input
    #    encoder_output = layers.Dense(2, activation="relu")(encoder_output)

    #    # Decoder.
    #    decoder_input = layers.Input(shape=(2,))
    #    decoder_output = layers.Dense(100, activation="sigmoid")(decoder_input)

    #    # Autoencoder.
    #    ae = ngdlmodels.AE(encoder_input, encoder_output, decoder_input, decoder_output, latent_dim=2)

    #    # Saving and loading.
    #    ae.save("test_ae")
    #    loaded_ae = ngdlmodels.load_ae_model("test_ae")

    #    self.assertTrue(are_models_same(ae, loaded_ae))


    #def test_vae_save_load(self):
    #    self.assertTrue(False, "Implement")


    #def test_tl_save_load(self):
    #    self.assertTrue(False, "Implement")


if __name__ == '__main__':
    unittest.main()
