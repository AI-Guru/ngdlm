"""
This module contains all next generation deep learning models.
"""

from keras.engine.training import Model
from keras import layers, optimizers, losses
from keras import backend as K
import numpy as np


class AE(Model):
    """ Autoencoder. This is a simple autoencoder consisting of an encoder and a decoder."""

    def __init__(self, encoder_input, encoder_output, decoder_input, decoder_output, latent_dim, activation="relu"):
        super(AE, self).__init__()

        self.latent_dim = latent_dim

        # Creating the encoder.
        self.encoder_input = encoder_input
        self.encoder_output = encoder_output
        self.encoder = Model(self.encoder_input, self.encoder_output, name="encoder")

        # Creating the decoder.
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output
        self.decoder = Model(self.decoder_input, self.decoder_output, name="decoder")

        # Creating the AE.
        inputs = self.encoder.inputs[0]
        outputs = self.decoder(self.encoder(inputs))
        self.model = Model(inputs, outputs, name='ae')


    def compile(
        self,
        optimizer,
        loss=None,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        **kwargs):

        assert "reconstruction_loss" not in kwargs, "Not expected to use reconstruction_loss in AE."

        self.model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        **kwargs):

        return self.model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)


    def fit_generator(
        self,
        generator,
        steps_per_epoch=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0):

        return self.model.fit_generator(generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)


    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None):

        return self.model.evaluate(x, y, batch_size, verbose, sample_weight, steps=None)

    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None):

        return self.model.predict(x, batch_size, verbose, steps)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()



class VAE(AE):
    """ Variational Autoencoder. This consists of an encoder and a decoder plus an interpolateable latent space. """

    def __init__(self, encoder_input, encoder_output, decoder_input, decoder_output, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Creating the encoder.
        self.encoder_input = encoder_input
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(encoder_output)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(encoder_output)
        z =layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        self.encoder_output = z
        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

        # Creating the decoder.
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output
        self.decoder = Model(self.decoder_input, self.decoder_output, name="decoder")

        # Creating the VAE.
        inputs = self.encoder.inputs[0]
        outputs = self.decoder(self.encoder(inputs)[2]) # This is z.
        self.model = Model(inputs, outputs, name="vae")


    def compile(
        self,
        optimizer,
        loss=None,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        reconstruction_loss="mse",
        **kwargs):

        assert loss == None, "Not expected to provide an explicit loss for VAE."

        # Inputs.
        inputs = self.encoder.inputs[0]
        inputs_dim = int(np.prod(inputs.shape.as_list()[1:]))

        # Outputs.
        z_mean = self.encoder.outputs[0]
        z_log_var = self.encoder.outputs[1]
        outputs = self.decoder(self.encoder(inputs)[2]) # This is z.

        # Define the loss.
        def vae_loss(loss_inputs, loss_outputs):

            # Flatten all to accept different dimensions.
            loss_inputs = K.flatten(loss_inputs)
            loss_outputs = K.flatten(loss_outputs)

            # Reconstruction loss.
            if reconstruction_loss == "mse":
                r_loss = losses.mse(loss_inputs, loss_outputs)
            elif reconstruction_loss == "binary_crossentropy":
                r_loss = losses.binary_crossentropy(loss_inputs, loss_outputs)
            else:
                raise Exception("Unexpected: " + str(reconstruction_loss))
            r_loss *= inputs_dim

            # kl loss.
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            # VAE loss.
            vae_loss = K.mean(r_loss + kl_loss)
            vae_loss /= inputs_dim
            return vae_loss

        # Compile model.
        loss = vae_loss
        self.model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class TL(Model):

    def __init__(self, base_input, base_output):
        super(TL, self).__init__()
        print("Initializing TL")

        # Creating the base-model.
        self.base_input = base_input
        self.base_output = base_output
        self.base = Model(self.base_input, self.base_output, name="base_model")

        # TODO create siamese net


class TLAE(Model):

    def __init__(self, encoder, decoder, latent_dim):
        print("Initializing TAE")


class TLVAE(Model):

    def __init__(self, encoder, decoder, latent_dim):
        print("Initializing TVAE")
