from .ae import AE
from .helpers import append_to_filepath
from tensorflow.keras import losses
from tensorflow.keras import backend as K


class CAE(AE):
    """
    Contractive Autoencoder. This is a autoencoder consisting of an encoder and a decoder. It has a special loss.
    http://www.icml-2011.org/papers/455_icmlpaper.pdf
    """


    def __init__(
        self,
        encoder=None, decoder=None, autoencoder=None):
        super(CAE, self).__init__(encoder=encoder, decoder=decoder, autoencoder=autoencoder)


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
        """
        Compiles the CAE.

        Additionally to the default functionality of *compile*, it adds the contractive loss.
        This loss takes the provided loss and adds a penalty term.
        """

        self.loss = loss

        def contractive_loss(y_pred, y_true):

            # Get the base_loss.
            if isinstance(self.loss, str):
                base_loss = losses.get(self.loss)(y_pred, y_true)
            else:
                base_loss = self.loss(y_pred, y_true)

            # Get the contractive loss.
            encoder_output = self.encoder.layers[-1]
            weigths = K.variable(value=encoder_output.get_weights()[0])
            weigths = K.transpose(weigths)  # N_hidden x N
            h = encoder_output.output
            dh = h * (1 - h)
            contractive = lam * K.sum(dh**2 * K.sum(weigths**2, axis=1), axis=1)

            return base_loss + contractive

        # Compile model.
        loss = contractive_loss
        self.autoencoder.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, **kwargs)
