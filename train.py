import datetime
import os
import urllib

import torch
import torchaudio
from torch.utils.data import Dataset

from voicebox_pytorch import (
    VoiceBox,
    EncodecVoco,
    ConditionalFlowMatcherWrapper,
    VoiceBoxTrainer
)
from audiolm_pytorch import HubertWithKmeans

from einops import rearrange
import argparse,random

from glob import glob
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, datasetpath, val_flag=False):
        super().__init__()
        files = glob(f"{datasetpath}/**/*.wav", recursive=True)  + \
                glob(f"{datasetpath}/**/*.flac", recursive=True) + \
                glob(f"{datasetpath}/**/*.mp3", recursive=True)

        self.files = files
        self.val_flag = val_flag

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = None
        while True:
            filename = self.files[idx] if filename is None else random.choice(self.files)

            wav, sr = torchaudio.load(filename)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # for train, we use 5s clip, for val we use full clip
            if self.val_flag:
                if wav.shape[1] > 20*sr:
                    wav = wav[:,0:20*sr]
                if sr!=24000:
                    wav = torchaudio.transforms.Resample(sr,24000)(wav)
                return rearrange(wav, '1 ... -> ...')
            
            # if (wav.shape[1] > 24*sr) or (wav.shape[1] < 4*sr):
            #     continue # too long or too short
            if wav.shape[1] > 5*sr:
                offset = random.randint(0, wav.shape[1] - 5*sr)
                wav = wav[:,offset:offset+5*sr]

            if sr!=24000:
                wav = torchaudio.transforms.Resample(sr,24000)(wav)
            wav = rearrange(wav, '1 ... -> ...')

            return wav





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="dir of train data")
    parser.add_argument("--ckpt", type=str, default=None, help="")
    parser.add_argument("--logs", type=str, default='./logs', help="")
    parser.add_argument("--bs", type=int, default=32, help="")
    parser.add_argument("--grad_accum_every", type=int, default=1, help="")
    args = parser.parse_args()
    print(args)

    accelerator = "cuda"
    hubert_ckpt_path = f"./checkpoints/hubert_base_ls960.pt"
    hubert_quantizer_path = f"./checkpoints/hubert_base_ls960_L9_km2000_expresso.bin"

    wav2vec = HubertWithKmeans(
        checkpoint_path = hubert_ckpt_path,
        kmeans_path = hubert_quantizer_path,
        target_sample_hz = 24_000,
    )
    wav2vec = wav2vec.to(accelerator)
    wav2vec.eval()


    model = VoiceBox(
        dim = 512,
        dim_cond_emb = 512,
        audio_enc_dec = EncodecVoco(),
        num_cond_tokens = 2001,
        depth = 12,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        attn_qk_norm = False,
        num_register_tokens = 0,
        use_gateloop_layers = False,
    )

    cfm_wrapper = ConditionalFlowMatcherWrapper(
        voicebox = model,
        cond_drop_prob = 0.2,
        wav2vec = wav2vec
    )
    # cfm_wrapper.load_state_dict(torch.load("./checkpoints/voicebox_small.pt", map_location = 'cpu'))
    if args.ckpt:
        cfm_wrapper.load_state_dict(torch.load(args.ckpt, map_location = 'cpu'))

    cfm_wrapper = cfm_wrapper.to(accelerator)

    print("Parameters: ", sum(p.numel() for p in cfm_wrapper.parameters()))


    audio_dataset_train = AudioDataset(args.train)
    audio_dataset_val = AudioDataset(args.train, val_flag=True)

    trainer = VoiceBoxTrainer(
        cfm_wrapper=cfm_wrapper,
        batch_size=args.bs,
        grad_accum_every = args.grad_accum_every,
        dataset_train = audio_dataset_train,
        dataset_val = audio_dataset_val,
        num_train_steps = 100000,
        num_warmup_steps = 5000,
        num_epochs = None,
        lr = 2e-4,
        initial_lr = 1e-5,
        wd = 0.,
        max_grad_norm = 0.2,
        valid_frac = 0,
        random_split_seed = 42,
        log_every = 10,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = args.logs,
        force_clear_prev_results = None,
        num_workers = 16,
        keep_save_count = 2
    )

    trainer.train()