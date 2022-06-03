import tensorflow as tf
from keras import Sequential
from kapre import (
    STFT,
    Magnitude,
    ApplyFilterbank,
    MagnitudeToDecibel
)
from kapre.composed import get_melspectrogram_layer


CLIP_DURATION = 5
SAMPLING_RATE = 8000
NUM_CHANNELS = 2
INPUT_SHAPE = ((CLIP_DURATION * SAMPLING_RATE), NUM_CHANNELS)


def create_model_with_audio_unit():
    # create all Kapre layers
    stft_layer = STFT(
        n_fft=1024,
        hop_length=512,
        input_data_format="channels_last",
        output_data_format="channels_last",
        input_shape=INPUT_SHAPE
    )
    magnitude_layer = Magnitude()
    filterbank_kwargs = {
        'sample_rate': SAMPLING_RATE,
        'n_freq': 513,
        'n_mels': 64,
        'f_min': 0.0,
        'f_max': 8000,
    }
    apply_filterbank_layer = ApplyFilterbank(type="mel", filterbank_kwargs=filterbank_kwargs)
    magnitude_to_decibel_layer = MagnitudeToDecibel()

    # add Kapre layers to a Sequential model
    model = Sequential()
    model.add(stft_layer)
    model.add(magnitude_layer)
    model.add(apply_filterbank_layer)
    model.add(magnitude_to_decibel_layer)

    vgg_16 = tf.keras.applications.VGG16(
        input_shape=(77, 64, 2), weights=None, classes=2
    )

    model.add(vgg_16)
    model.summary()


def create_model_with_audio_unit_from_utility_function():
    mel_spectrogram_layer = get_melspectrogram_layer(
        n_fft=1024,
        hop_length=512,
        input_data_format="channels_last",
        output_data_format="channels_last",
        input_shape=INPUT_SHAPE,
        sample_rate=SAMPLING_RATE,
        n_mels=64,
        mel_f_min=0.0,
        mel_f_max=8000
    )

    model = Sequential()
    model.add(mel_spectrogram_layer)

    vgg_16 = tf.keras.applications.VGG16(
        input_shape=(77, 64, 2), weights=None, classes=2
    )

    model.add(vgg_16)
    model.summary()


if __name__ == "__main__":
    create_model_with_audio_unit_from_utility_function()


