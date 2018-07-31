import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers
import logging
import numpy as np

class TestModels(unittest.TestCase):


    def test_ae_functional(self):

        # Encoder.
        encoder_input = layers.Input(shape=(100,))
        encoder_output = encoder_input
        encoder_output = layers.Dense(2, activation="relu")(encoder_output)
        encoder = models.Model(encoder_input, encoder_output)

        # Decoder.
        decoder_input = layers.Input(shape=(2,))
        decoder_output = layers.Dense(100, activation="sigmoid")(decoder_input)
        decoder = models.Model(decoder_input, decoder_output)

        # Autoencoder.
        ae = ngdlmodels.AE(encoder=encoder, decoder=decoder)
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
        encoder = models.Model(encoder_input, encoder_output)

        # Decoder.
        decoder_input = layers.Input(shape=(2,))
        decoder_output = layers.Dense(20, activation="relu")(decoder_input)
        decoder_output = layers.Dense(100, activation="sigmoid")(decoder_output)
        decoder = models.Model(decoder_input, decoder_output)

        # Variational Autoencoder.
        vae = ngdlmodels.VAE(encoder=encoder, decoder=decoder, latent_dim=2)
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
        base = models.Model(base_model_input, base_model_output)

        # Triplet loss.
        tl = ngdlmodels.TL(base)
        tl.summary()


    def test_tl_sequential(self):

        # Base model.
        base_model = models.Sequential()
        base_model.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Triplet loss.
        tl = ngdlmodels.TL(base=base_model)
        tl.summary()


    def test_ae_save_load(self):

        # Encoder.
        encoder_input = layers.Input(shape=(100,))
        encoder_output = encoder_input
        encoder_output = layers.Dense(2, activation="relu")(encoder_output)
        encoder = models.Model(encoder_input, encoder_output)

        # Decoder.
        decoder_input = layers.Input(shape=(2,))
        decoder_output = layers.Dense(100, activation="sigmoid")(decoder_input)
        decoder = models.Model(decoder_input, decoder_output)

        # Autoencoder.
        ae = ngdlmodels.AE(encoder, decoder)
        self.assertTrue(self.are_models_same(ae, ae))

        # Saving and loading.
        ngdlmodels.save_model(ae, "test_ae")
        loaded_ae = ngdlmodels.load_ae_model("test_ae")

        self.assertTrue(self.are_models_same(ae, loaded_ae))


    def are_models_same(self, model1, model2):

        log= logging.getLogger( "SomeTest.testSomething" )
        weights1 = model1.model.get_weights()
        weights2 = model2.model.get_weights()

        return self.my_are_same(weights1, weights2)


    def my_are_same(self, e1, e2):

        if type(e1) == list and type(e1) == list:
            all_same = True

            if len(e1) != len(e2):
                return False

            for x1, x2 in zip(e1, e2):
                if self.my_are_same(x1, x2) == False:
                    all_same == False
                    break
            return all_same

        elif type(e1) == type(e2):
            return np.array_equal(e1, e2)

        else:
            return False

    #def test_vae_save_load(self):
    #    self.assertTrue(False, "Implement")


    #def test_tl_save_load(self):
    #    self.assertTrue(False, "Implement")


if __name__ == '__main__':
    #logging.basicConfig( stream=sys.stderr )
    #logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    #unittest.main()

    TestModels().test_ae_save_load()
