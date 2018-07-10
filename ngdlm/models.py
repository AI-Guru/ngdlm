from keras.engine.training import Model
from keras import layers, optimizers
from keras import backend as K

class AE(Model):
    """ Autoencoder. """

    def __init__(self, latent_dim, activation="relu"):
        print("Initializing AE")

        self.latent_dim = latent_dim


    def set_encoder(self, encoder_input, encoder_output):
        self.encoder_input = encoder_input
        self.encoder_output = layers.Dense(self.latent_dim, activation=self.activation)(encoder_output)
        self.encoder = Model(self.encoder_input, self.encoder_output, name="encoder")


    def set_decoder(self, decoder_input, decoder_output):
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output
        self.decoder = Model(self.decoder_input, self.decoder_output, name="decoder")
        #self.model = Model(self.encoder_input, self.decoder_output)


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

        inputs = self.encoder.inputs[0]
        outputs = self.decoder(self.encoder(inputs))
        self.model = Model(inputs, outputs, name='ae')
        self.model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)

        return self.model

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
    """ Variational Autoencoder. """


    def set_encoder(self, encoder_input, encoder_output):
        self.encoder_input = encoder_input

        # Latent layers. Mean, variation and reparametrization trick.
        z_mean = layers.Dense(self.latent_dim, name='z_mean', activation=self.activation)(encoder_output)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var', activation=self.activation)(encoder_output)
        z =layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        self.encoder_output = z
        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')


    def create_model(self):
        # Inputs.
        inputs = self.encoder.inputs[0]

        # Outputs.
        z_mean = self.encoder.outputs[0]
        z_log_var = self.encoder.outputs[1]
        outputs = self.decoder(self.encoder(inputs)[2]) # This is z.

        # Create the model.
        self.model = Model(inputs, outputs, name='vae_mlp')

        # TODO use other loss here. https://blog.keras.io/building-autoencoders-in-keras.html
        def vae_loss(loss_inputs, loss_outputs):
            # Reconstruction loss.
            if reconstruction_loss_type == "mse":
                reconstruction_loss = mse(loss_inputs, loss_outputs)
            elif reconstruction_loss_type == "binary_crossentropy":
                reconstruction_loss = binary_crossentropy(loss_inputs, loss_outputs)
            reconstruction_loss *= original_dim

            # kl loss.
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            # VAE loss.
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            return vae_loss


        # Compile model.
        self.model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=vae_loss)

        return self.model


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
# TODO use sampling from https://blog.keras.io/building-autoencoders-in-keras.html
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


class TAE(Model):

    def __init__(self, encoder, decoder, latent_dim):
        print("Initializing TAE")


class TVAE(Model):

    def __init__(self, encoder, decoder, latent_dim):
        print("Initializing TVAE")
