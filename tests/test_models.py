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


    def test_ae_sequential(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(2, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Autoencoder.
        ae = ngdlmodels.AE(encoder=encoder, decoder=decoder)


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


    def test_vae_sequential(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Variational Autoencoder.
        vae = ngdlmodels.VAE(encoder=encoder, decoder=decoder, latent_dim=2)


    def test_tl_functional(self):

        # Base model.
        base_model_input = layers.Input(shape=(100,))
        base_model_output = base_model_input
        base_model_output = layers.Dense(20, activation="relu")(base_model_output)
        base = models.Model(base_model_input, base_model_output)

        # Triplet loss.
        tl = ngdlmodels.TL(base)


    def test_tl_sequential(self):

        # Base model.
        base_model = models.Sequential()
        base_model.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Triplet loss.
        tl = ngdlmodels.TL(base=base_model)


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
        ae = ngdlmodels.AE(encoder=encoder, decoder=decoder)
        self.assert_models_same(ae, ae)

        # Saving and loading.
        ae.save("test_ae")
        loaded_ae = ngdlmodels.load_ae_model("test_ae")

        self.assert_models_same(ae, loaded_ae)


    def test_vae_save_load(self):

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
        vae = ngdlmodels.VAE(encoder=encoder, decoder=decoder, latent_dim=2)
        self.assert_models_same(vae, vae)

        # Saving and loading.
        vae.save("test_vae")
        loaded_vae = ngdlmodels.load_vae_model("test_vae")

        self.assert_models_same(vae, loaded_vae)


    def assert_models_same(self, model1, model2):

        assert type(model1) is type(model2)

        if type(model1) is ngdlmodels.AE  and type(model2) is ngdlmodels.AE:
            assert self.are_weights_same(model1.encoder.get_weights(), model2.encoder.get_weights())
            assert self.are_weights_same(model1.decoder.get_weights(), model2.decoder.get_weights())
            assert self.are_weights_same(model1.autoencoder.get_weights(), model2.autoencoder.get_weights())
        elif type(model1) is ngdlmodels.VAE  and type(model2) is ngdlmodels.VAE:
            assert model1.latent_dim == model2.latent_dim
            assert self.are_weights_same(model1.encoder.get_weights(), model2.encoder.get_weights())
            assert self.are_weights_same(model1.decoder.get_weights(), model2.decoder.get_weights())
            assert self.are_weights_same(model1.autoencoder.get_weights(), model2.autoencoder.get_weights())
        else:
            raise Exception("Unexpected: " + type(model1) + " " + type(model2))

    def are_weights_same(self, e1, e2):

        if type(e1) == list and type(e1) == list:
            all_same = True

            if len(e1) != len(e2):
                return False

            for x1, x2 in zip(e1, e2):
                if self.are_weights_same(x1, x2) == False:
                    all_same == False
                    break
            return all_same

        elif type(e1) == type(e2):
            return np.array_equal(e1, e2)

        else:
            return False


    def test_ae_predictions(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(2, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Autoencoder.
        ae = ngdlmodels.AE(encoder=encoder, decoder=decoder)

        prediction = ae.predict(np.random.random((1, 100)))
        assert prediction.shape == (1, 100), "Unexpected shape " + str(prediction.shape)

        prediction = ae.predict_reconstruct_from_samples(np.random.random((1, 100)))
        assert prediction.shape == (1, 100), "Unexpected shape " + str(prediction.shape)

        prediction = ae.predict_embed_samples_into_latent(np.random.random((1, 100)))
        assert prediction.shape == (1, 2), "Unexpected shape " + str(prediction.shape)

        prediction = ae.predict_reconstruct_from_latent(np.random.random((1, 2)))
        assert prediction.shape == (1, 100), "Unexpected shape " + str(prediction.shape)


    def test_vae_predictions(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Variational Autoencoder.
        vae = ngdlmodels.VAE(encoder=encoder, decoder=decoder, latent_dim=2)

        prediction = vae.predict(np.random.random((1, 100)))
        assert prediction.shape == (1, 100), "Unexpected shape " + str(prediction.shape)

        prediction = vae.predict_reconstruct_from_samples(np.random.random((1, 100)))
        assert prediction.shape == (1, 100), "Unexpected shape " + str(prediction.shape)

        prediction = vae.predict_embed_samples_into_latent(np.random.random((1, 100)))
        assert prediction.shape == (1, 2), "Unexpected shape " + str(prediction.shape)

        prediction = vae.predict_reconstruct_from_latent(np.random.random((1, 2)))
        assert prediction.shape == (1, 100), "Unexpected shape " + str(prediction.shape)


if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    unittest.main()
