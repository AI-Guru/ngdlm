from collections import defaultdict
from .helpers import append_to_filepath
from .helpers import euclidean_loss
from .helpers import cosine_loss
from .helpers import compute_latent_extremum
from keras.engine.training import Model
from keras import layers
from keras import backend as K
import numpy as np
import random


class TL(Model):
    """
    Triplet-Loss trained Neural Network.

    https://arxiv.org/abs/1503.03832
    """

    def __init__(self, base=None, siamese=None):
        super(TL, self).__init__()

        # Store the base model.
        assert (base != None)
        self.base = base

        # For loading.
        if base != None and siamese != None:
            self.base = base
            self.siamese = siamese
            self.latent_dim = self.base.outputs[0].shape[1]
            return

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
        self.siamese = Model([input_anchor, input_positive, input_negative], output, name="triplet_model")


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
        """
        Compiles the TL.

        Additionally to the default functionality of *compile*, it adds the triplet-loss.
        In order to do so you have to provide it via the parameter *triplet_loss*.

        The VAE loss is similar to

        >>> vae_loss = max(0.0, pos_dist - neg_dist + alpha)

        See the literature for details.

        Additional args:
            triplet_loss (string): The base-loss for the triplet-loss. Values are either *euclidean* for euclidean norm or *cosine* for cosine similarity.

        """
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

        self.siamese.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)


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
        """
        This is basically the same as in vanilla Keras.

        Additional args:
            minibatch_size (int): The model internally does some sampling. The *minibatch_size* specifies how many candidates to use in order to create a triplet for training.
        """

        assert minibatch_size != None, "ERROR! Must provide 'minibatch_size'."
        assert steps_per_epoch != None, "ERROR! Must provide 'steps_per_epoch'."
        assert validation_steps != None, "ERROR! Must provide 'validation_steps'."

        y_dummy = np.zeros((batch_size, self.latent_dim * 3))

        # Template generator.
        def triplet_loss_generator(x_generator, y_generator, model, sampling):

            # Get the classes.
            classes = sorted(list(set(y_generator)))

            # Sort by classes for easy indexing.
            class_indices = defaultdict(list)
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
        training_generator = triplet_loss_generator(x, y, batch_size, self.siamese)
        if validation_data != None:
            validation_generator = triplet_loss_generator(validation_data[0], validation_data[1], batch_size, self.siamese)
        else:
            validation_generator = None

        # Create the history.
        history_keys = ["loss", "val_loss"]
        history = defaultdict(list)

        # Training the model
        for epoch in range(epochs):

            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + "...")

            # Generating data for training.
            training_input, training_output = next(training_generator)
            if validation_generator != None:
                validation_input, validation_output = next(validation_generator)

            model_history = self.siamese.fit(
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
        """
        Coming soon...
        """

        print("TODO: implement fit_generator!")

        raise Exception("Not implemented!")

        return self.siamese.fit_generator(generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)


    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None):
        """
        Evaluates the model. Same as vanilla Keras.
        """

        return self.siamese.evaluate(x, y, batch_size, verbose, sample_weight, steps=None)


    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None):
        """
        Does a prediction. Same as vanilla Keras.
        """

        return self.siamese.predict(x, batch_size, verbose, steps)


    def summary(self):
        """
        Provides a summary.
        """

        print("Basemodel:")
        self.base.summary()
        print("Siamese model:")
        self.siamese.summary()


    def save(self, path):
        """
        Saves the TL.

        This includes the whole Siamese Net plus the base-model.

        This code

        >>> tl.save("myae.h5")

        will create the files *tl.h5*, and *tl-base.h5*.

        """
        self.siamese.save(path)
        self.base.save(append_to_filepath(path, "-base"))
