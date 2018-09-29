import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers
import logging
import numpy as np
import glob
import os


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

        self.assert_load_save_fine(ae)


    def test_ae_sequential(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(2, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Autoencoder.
        ae = ngdlmodels.AE(encoder=encoder, decoder=decoder)

        self.assert_load_save_fine(ae)


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
        vae.compile(optimizer="rmsprop", loss="mse")

        self.assert_load_save_fine(vae)


    def test_vae_sequential(self):

        # Encoder.
        encoder = models.Sequential()
        encoder.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Decoder.
        decoder = models.Sequential()
        decoder.add(layers.Dense(100, activation="sigmoid", input_shape=(2,)))

        # Variational Autoencoder.
        vae = ngdlmodels.VAE(encoder=encoder, decoder=decoder, latent_dim=2)
        vae.compile(optimizer="rmsprop", loss="mse")

        self.assert_load_save_fine(vae)


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

        # Train model.
        x_train = np.ones((10, 100, 1024))
        y_train = np.zeros((10, 100, 1024))
        history = tdlstmae.fit(
            x_train, y_train,
            epochs=1,
            batch_size=16
            )

        self.assert_load_save_fine(tdlstmae)


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

        # Train model.
        x_train = np.ones((10, 100))
        y_train = np.zeros((10, 100))
        history = cae.fit(
            x_train, y_train,
            epochs=1,
            batch_size=16
            )

        self.assert_load_save_fine(cae)


    def test_tl_functional(self):

        # Base model.
        base_model_input = layers.Input(shape=(100,))
        base_model_output = base_model_input
        base_model_output = layers.Dense(20, activation="relu")(base_model_output)
        base = models.Model(base_model_input, base_model_output)

        # Triplet loss.
        tl = ngdlmodels.TL(base)

        self.assert_load_save_fine(tl)


    def test_tl_sequential(self):

        # Base model.
        base_model = models.Sequential()
        base_model.add(layers.Dense(20, activation="relu", input_shape=(100,)))

        # Triplet loss.
        tl = ngdlmodels.TL(base=base_model)

        self.assert_load_save_fine(tl)


    def assert_load_save_fine(self, model):

        # Model should be the same as itself.
        self.assert_models_same(model, model)

        # Path for the model.
        mapping = {
            ngdlmodels.AE: "ae",
            ngdlmodels.VAE: "vae",
            ngdlmodels.CAE: "cae",
            ngdlmodels.TDLSTMAE: "tdlstmae",
            ngdlmodels.TL: "tl",

        }
        model_path = "unittest-" + mapping[type(model)] + ".h5"

        # Save the model.
        model.save(model_path)

        # Load the model.
        loaded_model = ngdlmodels.load_model(model_path, type(model))

        # Loaded model should be the same as original model.
        self.assert_models_same(model, loaded_model)



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
        elif type(model1) is ngdlmodels.CAE  and type(model2) is ngdlmodels.CAE:
            assert self.are_weights_same(model1.encoder.get_weights(), model2.encoder.get_weights())
            assert self.are_weights_same(model1.decoder.get_weights(), model2.decoder.get_weights())
            assert self.are_weights_same(model1.autoencoder.get_weights(), model2.autoencoder.get_weights())
        elif type(model1) is ngdlmodels.TDLSTMAE  and type(model2) is ngdlmodels.TDLSTMAE:
            assert self.are_weights_same(model1.encoder.get_weights(), model2.encoder.get_weights())
            assert self.are_weights_same(model1.decoder.get_weights(), model2.decoder.get_weights())
            assert self.are_weights_same(model1.autoencoder.get_weights(), model2.autoencoder.get_weights())
        elif type(model1) is ngdlmodels.TL  and type(model2) is ngdlmodels.TL:
            assert model1.latent_dim == model2.latent_dim
            assert self.are_weights_same(model1.base.get_weights(), model2.base.get_weights())
            assert self.are_weights_same(model1.siamese.get_weights(), model2.siamese.get_weights())
        else:
            raise Exception("Unexpected: " + str(type(model1)) + " " + str(type(model2)))


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


    def test_gan(self):
        # Some parameters.
        latent_dim = 100
        input_shape = (28, 28, 1)

        # Generator.
        generator = models.Sequential()
        generator.add(layers.Dense(256, input_dim=latent_dim))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.Dense(512))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.Dense(1024))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.Dense(np.prod(input_shape), activation='tanh'))
        generator.add(layers.Reshape(input_shape))

        # Discriminator.
        discriminator = models.Sequential()
        discriminator.add(layers.Flatten(input_shape=input_shape))
        discriminator.add(layers.Dense(512))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dense(256))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dense(1, activation='sigmoid'))
        discriminator.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"])

        # Load and transform the dataset.
        #(x_input_train, _), (_, _) = mnist.load_data()
        #x_input_train = x_input_train / 127.5 - 1.
        #x_input_train = np.expand_dims(x_input_train, axis=3)
        x_input_train = np.random.random((1, 28, 28, 1))

        # Create the net and train.
        gan = ngdlmodels.GAN(generator=generator, discriminator=discriminator)
        gan.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=['accuracy'])
        history = gan.fit(x_input_train, epochs=1, batch_size=32, sample_interval=None)


    def tearDown(self):
        files_to_delete = glob.glob("unittest*")
        for file in files_to_delete:
            os.remove(file)



if __name__ == '__main__':
    unittest.main()
