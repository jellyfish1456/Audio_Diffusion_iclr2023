from sc_dataset import *

import argparse
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio
from transforms.transforms_wav import *
from torch.utils.data import Dataset
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

'''SC09 classifier arguments'''
#parser.add_argument("--data_path", default='datasets/speech_commands/test')
parser.add_argument("--data_path", default=r'/home/Audio_Diffusion_iclr2023/audio_models/ConvNets_SpeechCommands/datasets/speech_commands/train')
parser.add_argument("--urban_data_path", default=r'/home/Audio_Diffusion_iclr2023/datasets/urbansound8k_organized')
parser.add_argument("--num_per_class", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4, help='batch size')
parser.add_argument("--dataload_workers_nums", type=int, default=0, help='number of workers for dataloader')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


'''device setting'''
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
print('gpu id: {}'.format(args.gpu))


'''sc09 specified'''
SC09_CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')
urbansound8k = 'airconditioner, carhorn, childrenplaying, dogbark, drilling, engineidling, gunshot, jackhammer, siren, streetmusic'.split(', ')

class SC09Datasets(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, folder, transform=None, classes=urbansound8k, num_per_class=10):


        all_classes = [d for d in classes if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        #print(all_classes) # ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        # print(len(all_classes))
        print(all_classes)
        
        
        for c in classes[:-2]:
           assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        #print(class_to_idx) # {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
        for c in all_classes:
            if c not in class_to_idx:
                # class_to_idx[c] = 0
                class_to_idx[c] = len(classes) - 1

        
        
        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d)[:min(num_per_class, len(os.listdir(d)))]:
                path = os.path.join(d, f)
                data.append((path, target))
                
        print(len(data))  #一個各10個  ('/home/Audio_Diffusion_iclr2023/audio_models/ConvNets_SpeechCommands/datasets/speech_commands/train/nine/0ff728b5_nohash_1.wav', 9)
        
        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight




    
transform = Compose([LoadAudio(), FixAudioLength()])
test_dataset = SC09Datasets(folder=args.urban_data_path, transform=transform, num_per_class=args.num_per_class)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
print("done")


