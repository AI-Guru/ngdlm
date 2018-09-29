from .ae import AE
from .helpers import append_to_filepath
from keras.engine.training import Model
from keras import layers
from keras import losses
from keras import backend as K
import numpy as np


class VAE(AE):
    """
    Variational Autoencoder. This consists of an encoder and a decoder plus an interpolateable latent space.
    """

    def __init__(
        self,
        encoder=None, decoder=None, autoencoder=None,
        latent_dim=None):
        super(VAE, self).__init__(encoder=None, decoder=None)

        # Encoder and decoder must be provided.
        assert (encoder != None and decoder != None)

        # From loading.
        if encoder != None and decoder != None and autoencoder != None:
            self.encoder = encoder
            self.decoder = decoder
            self.autoencoder = autoencoder
            self.latent_dim = decoder.inputs[0].shape.as_list()[-1]
            return

        # Set the latent dimensions.
        self.latent_dim = latent_dim
        assert self.latent_dim != None

        # Encoder.
        encoder_input = encoder.inputs[0]
        encoder_output = encoder.outputs[0]
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(encoder_output)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(encoder_output)
        z =layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

        # Decoder.
        self.decoder = decoder

        # Creating the VAE.
        inputs = self.encoder.inputs[0]
        outputs = self.decoder(self.encoder(inputs)[2]) # This is z.
        self.autoencoder = Model(inputs, outputs, name="vae")


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
        """
        Compiles the VAE.

        Additionally to the default functionality of *compile*, it adds the VAE-loss.
        This loss takes the provided loss and interprets it as a reconstruction-loss.

        The VAE loss is similar to

        >>> vae_loss = mean(r_loss + kl_loss)

        See the literature for details.

        """

        self.loss = loss

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
            if isinstance(self.loss, str):
                r_loss = losses.get(self.loss)(loss_inputs, loss_outputs)
            else:
                r_loss = self.loss(loss_inputs, loss_outputs)

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
        self.autoencoder.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


    def predict_embed_samples_into_latent(self, x, batch_size=None, verbose=0, steps=None):

        return self.encoder.predict(x, batch_size, verbose, steps)[2]


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
