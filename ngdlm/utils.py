import numpy as np
import matplotlib.pyplot as plt


def render_history(history):
    plt.plot(history.history["loss"], label="loss")
    if "val_loss" in history.history.keys():
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    plt.close()


def render_image_reconstructions(model, x_input, n):
    assert len(x_input.shape) == 3 or len(x_input.shape) == 4, "Expected data to have 3 or 4 dimensions."

    plt.figure(figsize=(20, 4))

    decoded_images = model.predict(x_input[0:n])

    for i in range(n):

        # Original.
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_input[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstruction.
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.close()


def render_image_latent_space(decoder, number_of_samples, latent_dim_1=0, latent_dim_2=1, space_range=4):

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

    render_embeddings(embeddings, rows=number_of_samples, columns=number_of_samples)


def render_embeddings(embeddings, rows, columns, title=None):

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
            axis.imshow(embedding, cmap="gray")
            axis.axis('off')

            if title != None:
                figure.suptitle(title, fontsize=48)

            index += 1

    plt.show()
    plt.close()
