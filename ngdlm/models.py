"""
This module contains all next generation deep learning models.
"""

from keras.engine.training import Model
from keras import layers, optimizers, losses
from keras import backend as K
import numpy as np
import random


class AE(Model):
    """ Autoencoder. This is a simple autoencoder consisting of an encoder and a decoder."""


    def __init__(
        self,
        encoder=None, decoder=None):
        super(AE, self).__init__()

        # For calling this as a super-constructor.
        parameters = [encoder, decoder]
        if all(v is None for v in parameters):
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

        #return self.model.fit_generator(generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
        return self.model.fit_generator(
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

        return self.model.evaluate(x, y, batch_size, verbose, sample_weight, steps=None)


    def predict(self, x, batch_size=None, verbose=0, steps=None):

        return self.predict_reconstruct_from_samples(x, batch_size, verbose, steps)


    def predict_reconstruct_from_samples(self, x, batch_size=None, verbose=0, steps=None):

        return self.model.predict(x, batch_size, verbose, steps)


    def predict_embed_samples_into_latent(self, x, batch_size=None, verbose=0, steps=None):

        return self.encoder.predict(x, batch_size, verbose, steps)


    def predict_reconstruct_from_latent(self, x, batch_size=None, verbose=0, steps=None):

        return self.decoder.predict(x, batch_size, verbose, steps)


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()



class TDLSTMAE(AE):
    """ Time-Distributed-LSTM-Autoencoder. This is a autoencoder consisting of an encoder and a decoder. Both are wrapped into TimeDistributed and connected via LSTM. """


    def __init__(
        self,
        encoder=None, decoder=None):
        super(TDLSTMAE, self).__init__(encoder=None, decoder=None)

        # Encoder and decoder must be provided.
        assert (encoder != None and decoder != None)

        # Check preconditions.
        assert len(encoder.outputs) == 1
        assert len(decoder.inputs) == 1
        assert encoder.outputs[0].shape[2:] == decoder.inputs[0].shape[2:] , str(encoder.outputs[0].shape) + " " + str(decoder.inputs[0].shape)
        self.latent_dim = encoder.outputs[0].shape[1]

        # Getting the input-tensors and the LSTM-size.
        encoder_input = encoder.inputs[0]
        decoder_input = decoder.inputs[0]
        lstm_size = decoder_input.shape.as_list()[-1]

        # Time-distributed encoder.
        time_distributed_encoder_input = encoder_input
        time_distributed_encoder_output = layers.TimeDistributed(encoder)(time_distributed_encoder_input)
        time_distributed_encoder = Model(inputs=time_distributed_encoder_input, outputs=time_distributed_encoder_output)
        self.encoder = time_distributed_encoder

        # Time-distributed decoder with LSTM.
        time_distributed_decoder_input = decoder_input
        time_distributed_decoder_output = layers.LSTM(lstm_size, return_sequences=True)(time_distributed_decoder_input)
        time_distributed_decoder_output = layers.TimeDistributed(decoder)(time_distributed_decoder_output)
        time_distributed_decoder = Model(inputs=time_distributed_decoder_input, outputs=time_distributed_decoder_output)
        self.decoder = time_distributed_decoder

        # Autoencoder.
        autoencoder_input = encoder_input
        autoencoder_output = autoencoder_input
        autoencoder_output = time_distributed_encoder(autoencoder_output)
        autoencoder_output = time_distributed_decoder(autoencoder_output)
        self.model = autoencoder = Model(inputs=autoencoder_input, outputs=autoencoder_output)


class CAE(AE):
    """ Contractive Autoencoder. This is a autoencoder consisting of an encoder and a decoder. It has a special loss. """


    def __init__(
        self,
        encoder=None, decoder=None):
        super(CAE, self).__init__(encoder=encoder, decoder=decoder)


    def compile(
        self,
        optimizer,
        loss=None,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        lam = 1e-4,
        **kwargs):

        assert loss == "mse", "Expected 'mse' as loss."

        def contractive_loss(y_pred, y_true):



            mse = K.mean(K.square(y_true - y_pred), axis=1)

            encoder_output = self.encoder.layers[-1]

            W = K.variable(value=encoder_output.get_weights()[0])  # N x N_hidden
            W = K.transpose(W)  # N_hidden x N
            h = encoder_output.output
            dh = h * (1 - h)  # N_batch x N_hidden

            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

            return mse + contractive

        # Compile model.
        loss = contractive_loss
        self.model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


class VAE(AE):
    """ Variational Autoencoder. This consists of an encoder and a decoder plus an interpolateable latent space. """


    def __init__(
        self,
        encoder=None, decoder=None,
        latent_dim=None):
        super(VAE, self).__init__(encoder=None, decoder=None)

        # Encoder and decoder must be provided.
        assert (encoder != None and decoder != None)

        # Set the latent dimensions.
        assert latent_dim != None
        self.latent_dim = latent_dim

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

        assert loss == None, "Not expected to provide an explicit loss for VAE. Use 'reconstruction_loss'"

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
            elif reconstruction_loss == "categorical_crossentropy":
                r_loss = losses.categorical_crossentropy(loss_inputs, loss_outputs)
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


class TL(Model):

    def __init__(self, base=None):
        super(TL, self).__init__()
        print("Initializing TL")

        # Store the base model.
        assert (base != None)
        self.base = base

        # Get the latent dimension.
        assert len(self.base.outputs) == 1
        assert len(self.base.outputs[0].shape) == 2
        self.latent_dim = self.base.outputs[0].shape[1]

        # Get the input shape.
        input_shape = self.base.inputs[0].shape.as_list()[1:]

        # Create the anchor.
        input_anchor = layers.Input(shape=input_shape)
        output_anchor = input_anchor
        output_anchor = self.base(output_anchor)

        # Create the positive.
        input_positive = layers.Input(shape=input_shape)
        output_positive = input_positive
        output_positive = self.base(output_positive)

        # Create the negative.
        input_negative = layers.Input(shape=input_shape)
        output_negative = input_negative
        output_negative = self.base(output_negative)

        # Create a dummy output.
        output = layers.concatenate([output_anchor, output_positive, output_negative])

        # Create the model.
        self.model = Model([input_anchor, input_positive, input_negative], output, name="triplet_model")


    def compile(
        self,
        optimizer,
        loss=None,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        triplet_loss="mse",
        **kwargs):

        assert loss == None, "Not expected to provide an explicit loss for TL. Use 'triplet_loss'"

        self.triplet_loss = triplet_loss

        def triplet_loss_function(y_true, y_pred, alpha = 0.4):

            anchor = y_pred[:,0:self.latent_dim]
            positive = y_pred[:,self.latent_dim:self.latent_dim * 2]
            negative = y_pred[:,self.latent_dim * 2:self.latent_dim * 3]

            if triplet_loss == "euclidean":
                pos_dist = euclidean_loss(positive, anchor)
                neg_dist = euclidean_loss(negative, anchor)
            elif triplet_loss == "cosine":
                pos_dist = cosine_loss(positive, anchor)
                neg_dist = cosine_loss(negative, anchor)
            else:
                raise Exception("Unexpected: " + triplet_loss)

            basic_loss = pos_dist - neg_dist + alpha
            loss = K.maximum(basic_loss, 0.0)
            return loss

        loss = triplet_loss_function

        self.model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        minibatch_size=None,
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

        assert minibatch_size != None, "ERROR! Must provide 'minibatch_size'."
        assert steps_per_epoch != None, "ERROR! Must provide 'steps_per_epoch'."
        assert validation_steps != None, "ERROR! Must provide 'validation_steps'."

        y_dummy = np.zeros((batch_size, self.latent_dim * 3))

        # Template generator.
        def triplet_loss_generator(x_generator, y_generator, model, sampling):

            # Get the classes.
            classes = sorted(list(set(y_generator)))

            # Sort by classes for easy indexing.
            class_indices = {}
            for c in classes:
                class_indices[c] = []
            for index, c in enumerate(y_generator):
                class_indices[c].append(index)

            # Compute the complements.
            class_complements = {}
            for c in classes:
                class_complements[c] = [c2 for c2 in classes if c2 != c]

            # Generator loop.
            while True:

                x_input_anchors = []
                x_input_positives = []
                x_input_negatives = []

                # Generate a whole batch.
                for _ in range(batch_size):
                    anchor_class = random.choice(classes)
                    anchor_index = random.choice(class_indices[anchor_class])
                    anchor_input = x_generator[anchor_index]
                    #print("anchor_class", anchor_class)
                    anchor_latent = self.base.predict(np.expand_dims(anchor_input, axis=0))[0]

                    # Generate some positive candidates.
                    positive_candidates = []
                    while len(positive_candidates) < minibatch_size:
                        positive_class = anchor_class
                        positive_index = random.choice(class_indices[positive_class])
                        positive_input = x_generator[positive_index]
                        assert positive_class == y_generator[positive_index]
                        #print("positive_class", positive_class)
                        positive_candidates.append(positive_input)

                    # Find the farthest positive candidate.
                    positive_candidates = np.array(positive_candidates)
                    positive_latents = self.base.predict(positive_candidates)
                    positive_extremum = compute_latent_extremum(anchor_latent, positive_latents, "argmax", self.triplet_loss)
                    positive_input = positive_candidates[positive_extremum]

                    # Generate some negative candidates.
                    negative_candidates = []
                    while len(negative_candidates) < minibatch_size:
                        negative_class = random.choice(class_complements[anchor_class])
                        negative_index = random.choice(class_indices[negative_class])
                        negative_input = x_generator[negative_index]
                        assert negative_class == y_generator[negative_index]
                        #print("negative_class", negative_class)
                        negative_candidates.append(negative_input)

                    # Find the closest negative candidate.
                    negative_candidates = np.array(negative_candidates)
                    negative_latents = self.base.predict(negative_candidates)
                    negative_extremum = compute_latent_extremum(anchor_latent, negative_latents, "argmin", self.triplet_loss)
                    negative_input = negative_candidates[negative_extremum]

                    # Done.
                    x_input_anchors.append(anchor_input)
                    x_input_positives.append(positive_input)
                    x_input_negatives.append(negative_input)

                x_input_anchors = np.array(x_input_anchors)
                x_input_positives = np.array(x_input_positives)
                x_input_negatives = np.array(x_input_negatives)
                x_input = [x_input_anchors, x_input_positives, x_input_negatives]

                yield x_input, y_dummy

        # Create the generators.
        training_generator = triplet_loss_generator(x, y, batch_size, self.model)
        if validation_data != None:
            validation_generator = triplet_loss_generator(validation_data[0], validation_data[1], batch_size, self.model)
        else:
            validation_generator = None

        # Create the history.
        history_keys = [
            "loss", "val_loss"]
        history = {}
        for history_key in history_keys:
            history[history_key] = []

        # Training the model
        for epoch in range(epochs):

            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + "...")

            # Generating data for training.
            training_input, training_output = next(training_generator)
            if validation_generator != None:
                validation_input, validation_output = next(validation_generator)

            model_history = self.model.fit(
                training_input, training_output,
                validation_data=(validation_input, validation_output),
                epochs=1,
                steps_per_epoch=steps_per_epoch,
                verbose=0,
                validation_steps=validation_steps
            )

            # Update the history.
            for history_key in history_keys:
                history_value = model_history.history[history_key]
                history[history_key].append(history_value)
                print(history_key, history_value)

        return history


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

        print("TODO: implement fit_generator!")

        raise Exception("Not implemented!")

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
        self.base.summary()
        self.model.summary()



class TLAE(Model):

    def __init__(self, encoder, decoder, latent_dim):
        print("Initializing TAE")


class TLVAE(Model):

    def __init__(self, encoder, decoder, latent_dim):
        print("Initializing TVAE")


def euclidean_loss(left, right):

    distance = K.sum(K.square(left - right), axis=1)
    return distance


def cosine_loss(left, right):

    left = K.l2_normalize(left, axis=-1)
    right = K.l2_normalize(right, axis=-1)

    distance = K.constant(1.0) - K.batch_dot(left, right, axes=1)
    distance = K.squeeze(distance, axis=-1)

    return distance


def compute_latent_extremum(latent_sample, latent_samples, extremum_type, norm):
    distances = [compute_latent_distance(latent_sample, l, norm) for l in latent_samples]
    if extremum_type == "argmax":
        return np.argmax(distances)
    elif extremum_type == "argmin":
        return np.argmin(distances)
    else:
        raise Exception("Unexpected: " + str(extremum_type))


def compute_latent_distance(latent_sample1, latent_sample2, norm):
    if norm == "euclidean":
        distance = np.sum(np.square(latent_sample2 - latent_sample1))
        return distance
    #elif norm == "cosine":
    #    distance = np.sum(np.square(latent_sample2 - latent_sample1))
    #    return distance
    else:
        raise Exception("Unexpected norm: " + norm)
