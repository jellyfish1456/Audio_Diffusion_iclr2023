import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
from utils import *
from datasets.sc_dataset import *
import torch.nn.functional as F
from audio_models.ConvNets_SpeechCommands.create_model import *

if __name__ == "__main__":
    # from https://github.com/huawei-noah/Speech-Backbones/blob/main/DiffVC/train_enc.py
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_data_path", default=r'/home/Audio_Diffusion_iclr2023/audio_models/ConvNets_SpeechCommands/datasets/speech_commands/train')
    parser.add_argument("--valid_data_path", default=r'/home/Audio_Diffusion_iclr2023/audio_models/ConvNets_SpeechCommands/datasets/speech_commands/valid')
    parser.add_argument("--num_per_class", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2, help='batch size')
    parser.add_argument("--dataload_workers_nums", type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument('--save_path', default='chris_try')
    parser.add_argument("--classifier_type", type=str, choices=['advtr', 'vanilla'], default='vanilla')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    
    
    '''DDPM-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--ddpm_path', type=str, default='diffusion_models/DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=10, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', action='store_true', default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')
    
    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['PGD', 'FAKEBOB'], default='PGD')
    parser.add_argument('--defense', type=str, choices=['Diffusion', 'Diffusion-Spec', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'None'], default='Diffusion')
    args = parser.parse_args()
    
    
    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))
    
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)
    os.makedirs(args.save_path, exist_ok=True)
    
    '''load classifier'''
    classifier_path = r"/home/Audio_Diffusion_iclr2023/audio_models/ConvNets_SpeechCommands/audio_models/ConvNets_SpeechCommands/checkpoints/resnext29_8_64_sgd_plateau_bs2_lr1.0e-04_wd1.0e-02/best-acc-speech-commands-checkpoint.pth"
    Classifier = create_model(classifier_path)
    save_every = 1
    
    
    
    '''load dataset'''
    print('Initializing train data loaders...')
    transform = Compose([LoadAudio(), FixAudioLength()])
    train_dataset = SC09Dataset(folder=args.train_data_path, transform=transform, num_per_class=args.num_per_class)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, shuffle=True, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

    print('Initializing valid data loaders...')
    transform = Compose([LoadAudio(), FixAudioLength()])
    valid_dataset = SC09Dataset(folder=args.valid_data_path, transform=transform, num_per_class=args.num_per_class)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    
    
    '''preprocessing setting (if use acoustic features like mel-spectrogram)'''
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])
    
    '''defense setting'''
    from acoustic_system import AcousticSystem
    if args.defense == 'Diffusion':
        
        from diffusion_models.diffwave_sde import *
        Defender = RevDiffWave(args) ### !!!!!!!!!!!!!!!!!!! check, 它好像就有在做加noise
        # Defender is diffusion model
        defense_type = 'wave'
        AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=Defender, defense_type=defense_type)
        print('classifier model: {}'.format(Classifier._get_name()))
        print('classifier type: {}'.format(args.classifier_type))
        if args.defense == 'Diffusion':
            print('defense: {} with t={}'.format(Defender._get_name(), args.t))
    
    
    '''---------------------- freeze model parameters ---------------------- '''
    # for param in AS_MODEL.parameters():
    #     param.requires_grad = False
        
    #AS_MODEL.eval()


    print('Initializing models...')
    # 還沒
    #fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size).cuda()
    #model = FwdDiffusion(n_mels, channels, filters, heads, layers, kernel, dropout, window_size, dim).cuda()

    '''---------------------- Encoder ----------------------'''
    #print(AS_MODEL)

    '''How to calculte the whole parameters'''
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    model_parameters = filter(lambda p: p.requires_grad, AS_MODEL.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The total parameters are:",params)

  
    
    print('Initializing optimizers...')
    # !!!!!!!!!!!!!!!!!!!! 改
    learning_rate = 5e-4
    optimizer = torch.optim.AdamW(
        AS_MODEL.parameters(),
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ''' Loss setting'''
    # def get_loss():
    #     #x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    #     x_noisy, noise = RevDiffWave(args)
    #     noise_pred = model(x_noisy, t)
    #     return F.l1_loss(noise, noise_pred)
    
    
    
    torch.backends.cudnn.benchmark = True
    iteration = 0
    
    min_valid_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        
        correct_orig = 0
        correct_orig_denoised = 0
        total = 0
       
        
        # !!!!!!!!!!!!!!!!!!!! 改
        AS_MODEL.train()
        losses = []
        train_loss = 0.0
        #for batch in tqdm(train_loader, total=len(train_set)//batch_size):
        # AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=Defender, defense_type=defense_type)
        '''---------------------- Training ----------------------'''
        for batch in tqdm(train_dataloader, total=len(train_dataset)//args.batch_size):
            print('Start training.')
            waveforms = batch['samples']
            waveforms = torch.unsqueeze(waveforms, 1)
            targets = batch['target']
            
            waveforms = waveforms.cuda()
            targets = targets.cuda()
            # print(waveforms.shape)
            
            '''original audio'''
            pred_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
            
            
            '''denoised original audio'''
            if AS_MODEL.defense_type == 'wave':
                waveforms_defended, noise = AS_MODEL.defender(waveforms) # 將waveform 經過diffusion and classifier
                
  
             
            pred_defended = AS_MODEL(waveforms_defended, False).max(1, keepdim=True)[1].squeeze()                    
       
            
            '''waveform/spectrogram saving'''
            if args.save_path is not None:
    
                clean_path = os.path.join(args.save_path,'clean')
               
                if not os.path.exists(clean_path):
                    os.makedirs(clean_path)

                for i in range(waveforms.shape[0]):
                    
                    audio_id = str(total + i).zfill(3)

                    if AS_MODEL.defense_type == 'wave': 
                        audio_save(waveforms[i], path=clean_path, 
                                        name='{}_{}_clean.wav'.format(audio_id,targets[i].item()))
                        audio_save(waveforms_defended[i], path=clean_path, 
                                        name='{}_{}_clean_purified.wav'.format(audio_id,targets[i].item()))

                
            # !!!!!!!!!!!!!!!!!!!! 改
            AS_MODEL.zero_grad()
            loss = F.mse_loss(waveforms_defended, noise)
            loss.backward()
            # !!!!!!!!!!!!!!!!!!!! 改
            torch.nn.utils.clip_grad_norm_(AS_MODEL.parameters(), max_norm=1)
            optimizer.step()
            
             # Calculate Loss
            train_loss += loss.item()

            losses.append(loss.item())
            iteration += 1
            
            
            '''metrics output'''
            total += waveforms.shape[0]
            correct_orig += (pred_clean==targets).sum().item()
            correct_orig_denoised += (pred_defended==targets).sum().item()
            acc_orig = correct_orig / total * 100
            acc_orig_denoised = correct_orig_denoised / total * 100

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg) 
        with open(f'{args.save_path}/train_enc.log', 'a') as f:
            f.write(msg)
        losses = []
 
        if epoch % save_every > 0:
            continue

        AS_MODEL.eval()
        '''---------------------- Validating ----------------------'''
        valid_loss = 0.0
        for batch_valid in tqdm(valid_dataloader, total=len(valid_dataset)//args.batch_size):
            print('Start validating.')
            waveforms = batch_valid['samples']
            waveforms = torch.unsqueeze(waveforms, 1)
            targets = batch_valid['target']
            
            waveforms = waveforms.cuda()
            targets = targets.cuda()
            
            '''original audio'''
            pred_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
            
            # Forward Pass
            '''denoised original audio'''
            if AS_MODEL.defense_type == 'wave':
                waveforms_defended, noise = AS_MODEL.defender(waveforms) # 將waveform 經過diffusion and classifier
            pred_defended = AS_MODEL(waveforms_defended, False).max(1, keepdim=True)[1].squeeze()   
            
            # Find the Loss
            # !!!!!!!!!!!!!!!!!!!! 改
            #loss = model.compute_loss(mel_x, mel_y, mel_mask)
            loss = F.mse_loss(waveforms_defended, noise)
            
            # Calculate Loss
            valid_loss += loss.item()
            
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(valid_dataloader)}')
        
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            
            # Saving State Dict
            torch.save(AS_MODEL.state_dict(), 'chris_valid_try.pth')