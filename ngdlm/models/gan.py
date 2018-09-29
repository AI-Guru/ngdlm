from keras.engine.training import Model
from keras import layers
import numpy as np


class GAN(Model):
    """ Generative Adversarial Network (GAN). """

    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        assert generator != None
        assert discriminator != None
        assert discriminator.optimizer != None, "Discriminator must be compiled!"

        self.generator = generator
        self.discriminator = discriminator

        # Create the GAN.
        z_shape = generator.inputs[0].shape[1:]
        gan_input = layers.Input(shape=z_shape)
        gan_output = gan_input
        gan_output = self.generator(gan_output)
        self.discriminator.trainable = False
        gan_output = self.discriminator(gan_output)
        self.gan = Model(gan_input, gan_output)


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
        Compiles the model. Same as vanilla Keras.
        """

        self.gan.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)



    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        sample_interval=None, # TODO document!
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
        Trains the GAN.

        This is almost the same as in vanilla Keras.
        """

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Select a random batch of images
            idx = np.random.randint(0, x.shape[0], batch_size)
            imgs = x[idx]

            # Create some noise.
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images.
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Create some noise.
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (to have the discriminator label samples as valid).
            g_loss = self.gan.train_on_batch(noise, valid)
            if type(g_loss) == list:
                g_loss = g_loss[0]

            # Plot the progress.
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss), end="\r")

            # If at save interval => save generated image samples
            if sample_interval != None and epoch % sample_interval == 0:
                self.sample_images(epoch)


    def sample_images(self, epoch):
        """
        Samples images.
        """

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        #fig.savefig("images/%d.png" % epoch)
        plt.show()
        plt.close()


    def summary(self):
        """
        Provides a summary.
        """

        print("Generator:")
        self.generator.summary()
        print("Discriminator:")
        self.discriminator.summary()
        print("GAN:")
        self.gan.summary()


    def save(self, path):
        """
        Saves the GAN.

        This includes the whole autoencoder plus the encoder and the decoder.
        The encoder and decoder use the path plus a respective annotation.

        This code

        >>> ae.save("myae.h5")

        will create the files *myae.h5*, *myae-encoder.h5*, and *myae-decoder.h5*.

        """
        self.gan.save(path)
        self.generator.save(append_to_filepath(path, "-generator"))
        self.discriminator.save(append_to_filepath(path, "-discriminator"))
