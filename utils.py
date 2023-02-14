import torch
import torchaudio
import numpy as np

import librosa.display
import matplotlib.pyplot as plt
from typing import Union
import os
import time
from utils import *
from scipy.io import wavfile
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def spec_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.shape == (32, 32)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=x, 
                                   x_axis='ms', y_axis='mel', 
                                   sr=16000, n_fft=2048, 
                                   fmin=0, fmax=8000, 
                                   ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    if path is None:
        path = './_Spec_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'spec.png'
    fig.savefig(os.path.join(path, name))

def audio_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    assert x.ndim == 2 and x.shape[0] == 1

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'audio.wav'

    torchaudio.save(os.path.join(path,name), x, 16000) # default sample rate = 16000

def audio_save_as_img(x: Union[np.ndarray, torch.Tensor], path=None, name=None, color=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.ndim == 1

    fig = plt.figure(figsize=(21, 9), dpi=100)

    plt.plot(x,'-',color=color if color is not None else 'steelblue')

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'waveform.png'

    fig.savefig(os.path.join(path, name))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def save_audio(file_path, sampling_rate, audio):
    audio = np.clip(audio.detach().cpu().squeeze().numpy(), -0.999, 0.999)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))



class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path']
        if path:
            samples, sample_rate = librosa.load(path, sr=self.sample_rate)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

# def get_scheduler(
#     name: Union[str, SchedulerType],
#     optimizer: Optimizer,
#     num_warmup_steps: Optional[int] = None,
#     num_training_steps: Optional[int] = None,
#     num_cycles: int = 1,
#     power: float = 1.0,
# ):
#     """
#     Unified API to get any scheduler from its name.
#     Args:
#         name (`str` or `SchedulerType`):
#             The name of the scheduler to use.
#         optimizer (`torch.optim.Optimizer`):
#             The optimizer that will be used during training.
#         num_warmup_steps (`int`, *optional*):
#             The number of warmup steps to do. This is not required by all schedulers (hence the argument being
#             optional), the function will raise an error if it's unset and the scheduler type requires it.
#         num_training_steps (`int``, *optional*):
#             The number of training steps to do. This is not required by all schedulers (hence the argument being
#             optional), the function will raise an error if it's unset and the scheduler type requires it.
#         num_cycles (`int`, *optional*):
#             The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
#         power (`float`, *optional*, defaults to 1.0):
#             Power factor. See `POLYNOMIAL` scheduler
#         last_epoch (`int`, *optional*, defaults to -1):
#             The index of the last epoch when resuming training.
#     """
#     name = SchedulerType(name)
#     schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
#     if name == SchedulerType.CONSTANT:
#         return schedule_func(optimizer)

#     # All other schedulers require `num_warmup_steps`
#     if num_warmup_steps is None:
#         raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

#     if name == SchedulerType.CONSTANT_WITH_WARMUP:
#         return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

#     # All other schedulers require `num_training_steps`
#     if num_training_steps is None:
#         raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

#     if name == SchedulerType.COSINE_WITH_RESTARTS:
#         return schedule_func(
#             optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles
#         )

#     if name == SchedulerType.POLYNOMIAL:
#         return schedule_func(
#             optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power
#         )

#     return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)