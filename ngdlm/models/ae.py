from keras.engine.training import Model
from .helpers import append_to_filepath


class AE(Model):
    """
    Autoencoder. This is a simple autoencoder consisting of an encoder and a decoder.

    You can use the class like this:
    >>> encoder = ...
    >>> decoder = ...
    >>> ae = Autoencoder(encoder=encoder, decoder=decoder)
    >>> ae.compile(...)
    >>> ae.fit(...)

    """

    def __init__(
        self,
        encoder=None, decoder=None, autoencoder=None):
        super(AE, self).__init__()

        # For calling this as a super-constructor.
        parameters = [encoder, decoder]
        if all(v is None for v in parameters):
            return

        # From loading.
        if encoder != None and decoder != None and autoencoder != None:
            self.encoder = encoder
            self.decoder = decoder
            self.autoencoder = autoencoder
            return

        # Check preconditions.
        assert len(encoder.outputs) == 1
        assert len(decoder.inputs) == 1
        assert encoder.outputs[0].shape[1:] == decoder.inputs[0].shape[1:] , str(encoder.outputs[0].shape) + " " + str(decoder.inputs[0].shape)
        self.latent_dim = encoder.outputs[0].shape[1]

        self.encoder = encoder
        self.decoder = decoder

        # Creating the AE.
        inputs = self.encoder.inputs[0]
        outputs = self.decoder(self.encoder(inputs))
        self.autoencoder = Model(inputs, outputs, name='ae')


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
        Compiles the model.

        This is the same as compilation in Keras.

        """


        assert "reconstruction_loss" not in kwargs, "Not expected to use reconstruction_loss in AE."

        self.autoencoder.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


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
        """
        Trains the autoencoder.
        """

        return self.autoencoder.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)


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
        """
        Trains the autoencoder with a generator.
        """

        return self.autoencoder.fit_generator(
            generator, steps_per_epoch, epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch)


    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None):
        """
        Evaluates the autoencoder.
        """

        return self.autoencoder.evaluate(x, y, batch_size, verbose, sample_weight, steps=None)


    def predict(self, x, batch_size=None, verbose=0, steps=None):
        """
        Does a prediction. This is the same as :func:`~ngdlm.models.AE.predict_reconstruct_from_samples`
        """

        return self.predict_reconstruct_from_samples(x, batch_size, verbose, steps)


    def predict_reconstruct_from_samples(self, x, batch_size=None, verbose=0, steps=None):
        """
        Reconstructs samples.

        Samples are firstly mapped to latent space using the encoder.
        The resulting latent vectors are then mapped to reconstruction space via the decoder.
        """

        return self.autoencoder.predict(x, batch_size, verbose, steps)


    def predict_embed_samples_into_latent(self, x, batch_size=None, verbose=0, steps=None):
        """
        Embeds samples into latent space using the encoder.
        """

        return self.encoder.predict(x, batch_size, verbose, steps)


    def predict_reconstruct_from_latent(self, x, batch_size=None, verbose=0, steps=None):
        """
        Maps latent vectors to reconstruction space using the decoder.
        """

        return self.decoder.predict(x, batch_size, verbose, steps)


    def summary(self):
        """
        Provides a summary.
        """

        print("Encoder:")
        self.encoder.summary()
        print("Decoder:")
        self.decoder.summary()
        print("Autoencoder:")
        self.autoencoder.summary()


    def save(self, path):
        """
        Saves the autoencoder.

        This includes the whole autoencoder plus the encoder and the decoder.
        The encoder and decoder use the path plus a respective annotation.

        This code

        >>> ae.save("myae.h5")

        will create the files *myae.h5*, *myae-encoder.h5*, and *myae-decoder.h5*.

        """
        self.autoencoder.save(path)
        self.encoder.save(append_to_filepath(path, "-encoder"))
        self.decoder.save(append_to_filepath(path, "-decoder"))
