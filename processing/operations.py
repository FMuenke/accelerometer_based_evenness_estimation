import numpy as np
import scipy
from scipy import ndimage
from scipy.signal import butter, lfilter


def encode_signal(acc_raw):
    encoded_signal = ""
    for x in acc_raw:
        encoded_signal += "{},".format(x)
    encoded_signal = encoded_signal[:-1]
    return encoded_signal


def decode_signal(row):
    return np.fromstring(row["raw_accelerometer_signal"], dtype=np.float32, sep=",")


#### Aggergation ####

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))


def std(signal):
    return np.std(signal)


def p10(signal):
    return np.percentile(signal, 10)


def p90(signal):
    return np.percentile(signal, 90)


def sum_of_magnitudes(signal):
    return np.sum(np.abs(np.fft.fft(signal)))


def mean_of_magnitudes(signal):
    return np.mean(np.abs(np.fft.fft(signal)))


def max_of_fft(signal):
    return np.max(np.abs(np.fft.fft(signal)))


def get_aggregation(aggregation_id):
    aggregations_list = {
        "RMS": rms,
        "STD": std,
        "P10": p10,
        "P90": p90,
        "MOM": mean_of_magnitudes,
        "MFFT": max_of_fft,
        "MAX": np.max
    }
    return aggregations_list[aggregation_id]


#### Operations ####


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, param, fs=100, order=5):
    lowcut, highcut = param.split("/")
    lowcut, highcut = np.max([float(lowcut), 1]), np.min([float(highcut), 49])

    b, a = butter_bandpass(float(lowcut), float(highcut), fs, order=order)
    y = lfilter(b, a, data)
    return y


def average_filter(data, kernel_size):
    kernel_size = int(kernel_size)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel)


def ramp_filter(data, kernel_size):
    kernel_size = int(kernel_size)
    flank1 = np.arange(1, kernel_size)
    flank2 = np.arange(1, kernel_size)[::-1]
    kernel = np.concatenate([flank1, flank2])
    kernel = kernel / np.sum(kernel)
    return np.convolve(data, kernel)


def gaussian_filter(data, kernel_size):
    return ndimage.gaussian_filter1d(data, sigma=kernel_size)


class Operation:
    catalogue = {
        "avg": average_filter,
        "rmp": ramp_filter,
        "bnd": butter_bandpass_filter
    }

    def __init__(self, op_id: str):
        self.op_type, self.op_param = op_id.split("-")
    
    def __str__(self):
        return "{}{}".format(self.op_type, self.op_param)

    def compute(self, signal):
        return self.catalogue[self.op_type](signal, self.op_param)
