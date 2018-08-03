"""
Utility functions for NGDLM. General rendering and building models.
"""

import numpy as np
import matplotlib.pyplot as plt
from ngdlm import models as ngdlmodels
from keras import models, layers
from PIL import Image


def render_history(history, zero_limit=False, show=True, figure_path=None):
    """
    Renders a training history.

    Args:
        history: A Keras history object.

    Returns:
        None
    """

    if history == None:
        print("WARNING! No history provided!")
        return

    if type(history) != dict:
        history = history.history

    plt.plot(history["loss"], label="loss")
    if "val_loss" in history.keys():
        plt.plot(history["val_loss"], label="val_loss")
    if zero_limit == True:
        plt.ylim(ymin=0.0)
    plt.legend()
    if show == True:
        plt.show()
    if figure_path != None:
        plt.savefig(figure_path)
    plt.close()


def render_image_reconstructions(model, x_input, cmap="gray", image_size=None, show=True, figure_path=None):
    """
    Renders reconstructions as images.

    Takes an array of input samples, predicts the reconstructions using the model and renders them.

    Args:
        model (Model): A model for predicting the reconstructions.
        x_input (ndarray): A array of input samples.

    Returns:
        None
    """


    assert len(x_input.shape) == 3 or len(x_input.shape) == 4, "Expected data to have 3 or 4 dimensions."

    n = len(x_input)

    plt.figure(figsize=(20, 4))

    decoded_images = model.predict(x_input)

    for i in range(n):

        # Getting the images.
        image_original = x_input[i]
        image_reconstructed = decoded_images[i]

        # Optional resizing.
        if image_size != None:
            image_original = Image.fromarray(image_original)
            image_original = image_original.resize(image_size, Image.ANTIALIAS)
            image_original = np.array(image_original)
            image_reconstructed = Image.fromarray(image_reconstructed)
            image_reconstructed = image_reconstructed.resize(image_size, Image.ANTIALIAS)
            image_reconstructed = np.array(image_reconstructed)

        # Original.
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image_original, cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Original")

        # Reconstruction.
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image_reconstructed, cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Reconstruction")

    if show == True:
        plt.show()
    if figure_path != None:
        plt.savefig(figure_path)
    plt.close()


def render_image_latent_space(decoder, number_of_samples, latent_dim_1=0, latent_dim_2=1, space_range=4, cmap="gray", show=True, figure_path=None):

    latent_dim = decoder.inputs[0].shape[1]

    grid_x = np.linspace(-space_range, space_range, number_of_samples)
    grid_y = np.linspace(-space_range, space_range, number_of_samples)[::-1]

    embeddings = []
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros(latent_dim)
            z_sample[latent_dim_1] = xi
            z_sample[latent_dim_2] = yi
            z_sample = np.expand_dims(z_sample, axis=0)
            x_decoded = decoder.predict(z_sample)
            embedding = x_decoded[0]
            embeddings.append(embedding)
    embeddings = np.array(embeddings)

    render_embeddings(embeddings, rows=number_of_samples, columns=number_of_samples, cmap=cmap, show=show, figure_path=figure_path)


def render_embeddings(embeddings, rows, columns, title=None, cmap=None, show=True, figure_path=None):

    figure = plt.figure(figsize=(24, 24))
    figure.subplots_adjust(hspace=0.1, wspace=0.001)
    #figure.set_facecolor((0.9, 0.9, 1.0))

    index = 0
    for row in range(rows):
        for column in range(columns):
            embedding = embeddings[index]
            #embedding = Image.fromarray(embedding)
            #embedding = embedding.resize((embedding_size, embedding_size), Image.ANTIALIAS)
            #embedding = np.array(embedding)

            axis = figure.add_subplot(rows, columns, index + 1)
            axis.imshow(embedding, cmap=cmap)
            axis.axis('off')

            if title != None:
                figure.suptitle(title, fontsize=48)

            index += 1

    if show == True:
        plt.show()
    if figure_path != None:
        plt.savefig(figure_path)
    plt.close()


def render_encodings(encoder, x_input, y_output, show=True, figure_path=None):

    x_test_encoded = encoder.predict(x_input, batch_size=32)
    if len(encoder.outputs) == 3:
        x_test_encoded = x_test_encoded[0]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_output, cmap="inferno")
    plt.colorbar()
    if show == True:
        plt.show()
    if figure_path != None:
        plt.savefig(figure_path)
    plt.close()



def build_dense_ae(input_shape, latent_dim, hidden_units=[], hidden_activation="relu"):
    return _build_dense(type="ae", input_shape=input_shape, latent_dim=latent_dim, hidden_units=hidden_units, hidden_activation=hidden_activation)


def build_dense_vae(input_shape, latent_dim, hidden_units=[], hidden_activation="relu"):
    return _build_dense(type="vae", input_shape=input_shape, latent_dim=latent_dim, hidden_units=hidden_units, hidden_activation=hidden_activation)


def _build_dense(type, input_shape, latent_dim, hidden_units=[], hidden_activation="relu"):

    # Some useful variables.
    input_size = np.prod(input_shape)
    hidden_units_reverse = hidden_units[::-1]

    # Create the encoder.
    encoder_input = layers.Input(shape=input_shape)
    encoder_output = layers.Reshape((input_size,))(encoder_input)
    for hidden in hidden_units:
        encoder_output = layers.Dense(hidden, activation=hidden_activation)(encoder_output)
    encoder_output = layers.Dense(latent_dim, activation=hidden_activation)(encoder_output)
    encoder = models.Model(encoder_input, encoder_output)

    # Create the decoder.
    decoder_input = layers.Input(shape=(latent_dim,))
    decoder_output = decoder_input
    for hidden in hidden_units_reverse:
        decoder_output = layers.Dense(hidden, activation=hidden_activation)(decoder_output)
    decoder_output = layers.Dense(input_size, activation="sigmoid")(decoder_output)
    decoder_output = layers.Reshape(input_shape)(decoder_output)
    decoder = models.Model(decoder_input, decoder_output)

    # Create the autoencoder.
    if type == "ae":
        ae = ngdlmodels.AE(encoder, decoder)
        return ae
    elif type == "vae":
        vae = ngdlmodels.VAE(encoder, decoder, latent_dim=latent_dim)
        return vae
