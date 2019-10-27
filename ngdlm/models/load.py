from .ae import AE
from .vae import VAE
from .cae import CAE
from .tdlstmae import TDLSTMAE
from .tl import TL
from tensorflow.keras import models


def load_ae_model(path):
    """
    Loads an AE from a given path.
    """
    return load_model(path, AE)


def load_cae_model(path):
    """
    Loads an CAE from a given path.
    """
    return load_model(path, CAE)


def load_tdlstmae_model(path):
    """
    Loads an TDLSTMAE from a given path.
    """
    return load_model(path, TDLSTMAE)


def load_vae_model(path):
    """
    Loads an VAE from a given path.
    """
    return load_model(path, VAE)


def load_tl_model(path):
    """
    Loads an TL from a given path.
    """
    return load_model(path, TL)


def load_model(path, model_type):
    """
    Loads an model from a given path using a given type.
    """
    model = models.load_model(path)

    if model_type is AE:
        return AE(encoder=model.layers[1], decoder=model.layers[2], autoencoder=model)
    elif model_type is CAE:
        return CAE(encoder=model.layers[1], decoder=model.layers[2], autoencoder=model)
    elif model_type is TDLSTMAE:
        return TDLSTMAE(encoder=model.layers[1], decoder=model.layers[2], autoencoder=model)
    elif model_type is VAE:
        return VAE(encoder=model.layers[1], decoder=model.layers[2], autoencoder=model)
    elif model_type is TL:
        return TL(base=model.layers[3], siamese=model)
    else:
        for layer in model.layers:
            print(type(layer))
        raise Exception("Unexpected type: " + str(type))
