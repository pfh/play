
import numpy as np
from scipy.io import wavfile
from scipy import signal

def convolve(signal1, signal2):
    spec1 = np.fft.fft(signal1)
    spec2 = np.fft.fft(signal2)
    return np.fft.ifft(spec1*spec2).real

length = 2**20

data = np.random.standard_t(df=3.0, size=length) / 256.0
#data = np.clip(data, -1.0, 1.0)

r = np.absolute(np.arange(length)-length//2)
conv = 1./((r/15.)**2+1.)

print data
data = convolve(data,conv)
print data

#data = np.tanh(data)

wavfile.write('/tmp/sound.wav', 44100, data)

#play /tmp/sound.wav repeat 1000000