NGDLM
===============================

version number: 0.0.1-rc1
author: Dr. Tristan Behrens

Overview
--------

Next Generation Deep Learning Models (NGDLM) for Keras is here! 

We live in such great times. It is marvellous! You see, basically everyone can do Deep Learning today. This was impossible about a decade ago. Thanks to so many people and institutions so many people have a blast training and deploying Deep Neural Networks.

NGDLM is all about Deep Neural Networks that are beyond simple Feed-Forward Networks. Ever heard about Autoencoders, Generative Adversarial Nets, and Triplet Loss? Implementing those could easily end up in building pyramids. I know what I am talking about.

NGDLM is a toolkit that helps you to easily create and train Deep Neural Networks of the next generation. This includes:

* Autoencoders (AEs)
* Contractive Autoencoders (CAEs)
* Time-Distributed LSTM Autoencoders (TDLSTMAEs)
* Variational Autoencoders (VAEs)
* Triplet-Loss-Trained Nets (TLs)
* Generative Adversarial Nets (GANs)


Demo - Variational Autoencoders
------------

Variational Autoencoders are great! They facilitate unsupervised learning with any data. What do they do? They let you embed any data into latent space. And they let you generate data from that latent space. Both directions! And it works without labels. Do you know another good thing? VAEs latent space is interpolateable. 

The trouble with VAEs is their construction. You see, you have to create an encoder and a decoder. After that you have to glue them together while making sure that the encoder predicts Gaussian distributions. And finally you have to add the VAE-loss. Sounds complicated? Yes, it is. But with NGDLM those efforts are reduced to a minimum. See for yourself, how you can train a VAE with NGDLM:

```python
# Import NGDLM models.
from ngdlm import models as ngdlmodels

# Train- and validation-data.
x_input_train = ...
x_input_validate = ...

# Create the encoder.
encoder = ...

# Create the decoder.
decoder = ...

# Create the variational autoencoder.
vae = ngdlmodels.VAE(encoder, decoder, latent_dim=2)
vae.compile(optimizer='adadelta', reconstruction_loss="binary_crossentropy")

# Train.
print("Train...")
history = vae.fit(
        x_input_train, x_input_train,
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_data=(x_input_validate, x_input_validate)
    )

# Evaluate.
print("Evaluate...")
loss = vae.model.evaluate(x_input_test, x_input_test)
print("Loss:", loss)
```

So you just create the encoder and the decoder. NGDLM then turns both into a VAE!

Demo - Visualizing Variational Autoencoders
------------

```Python
# Import NGDLM utils.
from ngdlm import utils as ngdlutils

# Visualizing variational autoencoder.
print("Rendering history...")
ngdlutils.render_history(history)

print("Rendering reconstructions...")
ngdlutils.render_image_reconstructions(vae, x_input_train[0:10])

print("Rendering latent-space...")
ngdlutils.render_image_latent_space(vae.decoder, 10)

print("Rendering encodings...")
ngdlutils.render_encodings(vae.encoder, x_input_test, y_output_test)
```

Demo - Triplet-Loss Training
------------

Let me provide another example... Remember [FaceNet](https://arxiv.org/abs/1503.03832)? Right, Google's Neural Net that takes photos of faces and embeds them into a vector. Yes, like word-embeddings. What they did is something huge. They trained with Triplet-Loss. You create a neural network that maps faces to embeddings. You then use three copies of that network and combine them into one huge Neural Net. And on top of that you than add the Triplet-Loss, which minimizes the distance between samples of the same class and maximizes the disctance between samples of different classes. Sounds like a huge effort implementing it? True!

TODO Sampling strategy...

Let me show you how this looks with NGDLM:

```python
# Import NGDLM models.
from ngdlm import models as ngdlmodels

# Train- and test-data.
x_input_train, y_output_train = ...
x_input_test, y_output_test = ...

# Triplet loss.
latent_dim = 8

# Create the base-model.
base = ...

# Create the triplet loss model.
tl = ngdlmodels.TL(base)
tl.compile(optimizer="rmsprop", triplet_loss="euclidean")

# Train.
print("Train...")
history = tl.fit(
        x_input_train, y_output_train,
        epochs=1000,
        batch_size=128,
        steps_per_epoch=1000,
        minibatch_size=10,
        shuffle=True,
        validation_data=(x_input_validate, y_output_validate),
        validation_steps=500
    )

# Evaluate.
print("Evaluate...")
loss = ae.model.evaluate(x_input_test, x_input_test)
print("Loss:", loss)
```

Demo - Visualizing Triplet-Loss
------------


```python
# Import NGDLM utils.
from ngdlm import utils as ngdlutils

# Visualizing triplet-loss.
print("Rendering history...")
ngdlutils.render_history(history)

print("Rendering encodings...")
ngdlutils.render_encodings(tl.base, x_input_train, y_output_train)
ngdlutils.render_encodings(tl.base, x_input_test, y_output_test)
```



Demo - Generative Adversarial Nets
-------------------------------------

```Python
# Import NGDLM models.
from ngdlm import models as ngdlmodels

# Some parameters.
latent_dim = 100
input_shape = (28, 28, 1)

# Generator.
generator = ...

# Discriminator.
discriminator = ...

# Load and transform the dataset.
x_input_train = ...

# Create the net and train.
gan = ngdlmodels.GAN(generator=generator, discriminator=discriminator)
gan.compile(
    optimizer=optimizers.Adam(0.0002, 0.5),
    loss="binary_crossentropy",
    metrics=['accuracy'])
gan.summary()
history = gan.fit(x_input_train, epochs=30000, batch_size=32, sample_interval=200)
```

Installation / Usage
--------------------

To install use pip:

    $ pip install git+https://github.com/ai-guru/ngdlm.git


Or clone the repo:

    $ git clone https://github.com/ai-guru/ngdlm.git
    $ python setup.py install
