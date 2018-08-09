from keras import models, layers
from ngdlm import models as ngdlmodels
from ngdlm import utils as ngdlutils
from keras.datasets import mnist
import numpy as np

# Train- and test-data.
(x_input_train, y_output_train), (x_input_validate, y_output_validate) = mnist.load_data()
x_input_train = x_input_train.astype("float32") / 255.0
x_input_validate = x_input_validate.astype("float32") / 255.0

# Triplet loss.
latent_dim = 8

# Create the base-model.
base_input = layers.Input(shape=(28, 28))
base_output = base_input
base_output = layers.Flatten()(base_output)
base_output = layers.Dense(512, activation="relu")(base_output)
base_output = layers.Dense(256, activation="relu")(base_output)
base_output = layers.Dense(128, activation="relu")(base_output)
base_output = layers.Dense(latent_dim)(base_output)
base = models.Model(base_input, base_output)

# Create the triplet loss model.
tl = ngdlmodels.TL(base)
tl.compile(optimizer="rmsprop", triplet_loss="euclidean")
tl.summary()

# Train.
print("Train...")
history = tl.fit(
        x_input_train, y_output_train,
        epochs=100,
        batch_size=128,
        steps_per_epoch=1000,
        minibatch_size=10,
        shuffle=True,
        validation_data=(x_input_validate, y_output_validate),
        validation_steps=500
    )

# Visualizing triplet-loss.
print("Rendering history...")
ngdlutils.render_history(history, show=False, figure_path="tl-history.png")

print("Rendering encodings...")
ngdlutils.render_encodings(tl.base, x_input_train, y_output_train, show=False, figure_path="tl-encodings-train.png")
ngdlutils.render_encodings(tl.base, x_input_validate, y_output_validate, show=False, figure_path="tl-encodings-validate.png")
