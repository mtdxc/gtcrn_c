import os
import soundfile as sf
from librosa import stft, istft
import os
import time
import onnxruntime
import numpy as np


## load data
in_name = 'test'
mix, fs = sf.read(in_name + '.wav', dtype='float32')
assert fs == 16000
## inference
win = np.hanning(512)**0.5
#for i in range(0, 512):
#    print(win[i])
input = stft(mix, n_fft=512, hop_length=256, win_length=512, window=win, center='False')
print('stft:', input.dtype, input.shape)
areal = np.expand_dims(np.real(input), axis=-1)
aimag = np.expand_dims(np.imag(input), axis=-1)
#complex[257, 611, 1, 2]
acmpx = np.stack((areal, aimag), axis=-1)
print('cmpx:', acmpx.dtype, acmpx.shape)
#[1, 257, 611, 2]
inputs = np.transpose(acmpx, axes=(2, 0, 1, 3))
print('inputs:', inputs.dtype, inputs.shape)
session = onnxruntime.InferenceSession('model/gtcrn.onnx', None, providers=['CPUExecutionProvider'])
conv_cache = np.zeros([2, 1, 16, 16, 33],  dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 16],  dtype="float32")
inter_cache = np.zeros([2, 1, 33, 16],  dtype="float32")
outputs = []
for i in range(inputs.shape[-2]):
        out_i, conv_cache, tra_cache, inter_cache = session.run([], {'mix': inputs[..., i:i+1, :],'conv_cache': conv_cache,'tra_cache': tra_cache,'inter_cache': inter_cache})
        #if(i == 0):
            #print(inputs[..., i:i+1, :])
        #print(i, "output_i", (np.array(out_i)).shape)
        outputs.append(out_i)
print((np.array(outputs)).shape)
outputs = np.concatenate(outputs, axis=2)
enhanced = istft(outputs[...,0] + 1j * outputs[...,1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5, center='False')
sf.write('enh_' + in_name + '.wav', enhanced.squeeze(), 16000)

