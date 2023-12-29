import datetime
import os
import urllib

import torch
import torchaudio

from voicebox_pytorch import (
    VoiceBox,
    EncodecVoco,
    ConditionalFlowMatcherWrapper
)
from audiolm_pytorch import HubertWithKmeans

from einops import rearrange
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ref", type=str, required=True, help="ref wav")
parser.add_argument("--tgt", type=str, required=True, help="the target wav, which will be converted")
parser.add_argument("--ckpt", type=str, default="./checkpoints/voicebox_small.pt")
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
    cond_drop_prob = 0.2
)
cfm_wrapper.load_state_dict(torch.load(args.ckpt, map_location = 'cpu'))
cfm_wrapper = cfm_wrapper.to(accelerator)

print("Parameters: ", sum(p.numel() for p in cfm_wrapper.parameters()))



wave, sr = torchaudio.load(args.tgt)
print(f'load target audio sr is {sr}')
if sr!=24000:
    wave = torchaudio.transforms.Resample(sr,24000)(wave)
wave = wave.to(accelerator)
semantic_token_ids = (wav2vec(wave) + 1).to(accelerator) # offset for the padding token

# print(semantic_token_ids)
print('semantic_token_ids shape', semantic_token_ids.shape)
print('tgt wave shape', wave.shape)
print('')
# exit()

# unconditional generation

start_date = datetime.datetime.now()

output_wave = cfm_wrapper.sample(
    semantic_token_ids = semantic_token_ids,
    steps = 32,
    cond_scale = 1.3
)
elapsed_time = (datetime.datetime.now() - start_date).total_seconds()

output_wave = rearrange(output_wave, "1 1 n -> 1 n")
output_duration = float(output_wave.shape[1]) / 24000
realtime_mult = output_duration / elapsed_time

print(f"Generated sample of duration {output_duration:0.2f}s in {elapsed_time}s ({realtime_mult:0.2f}x realtime)")

torchaudio.save(f"{args.tgt.replace('.wav','')}_uncondition.wav",output_wave.cpu(),24000)


# Infilled
cond_wave, sr = torchaudio.load(args.ref)
if sr!=24000:
    cond_wave = torchaudio.transforms.Resample(sr,24000)(cond_wave)
cond_wave = cond_wave.to(accelerator)
cond_semantic_token_ids = (wav2vec(cond_wave) + 1).to(accelerator) # offset for the padding token

cond = torch.cat([cond_wave, wave], dim = -1)
cond_mask_copy = torch.zeros_like(cond_semantic_token_ids, dtype = torch.bool)
cond_mask_infill = torch.ones_like(semantic_token_ids, dtype = torch.bool)
cond_mask = torch.cat([cond_mask_copy, cond_mask_infill], dim = -1).to(accelerator)

start_date = datetime.datetime.now()

infilled_wave = cfm_wrapper.sample(
    cond = cond,
    cond_mask = cond_mask,
    semantic_token_ids = torch.cat([cond_semantic_token_ids, semantic_token_ids], dim = -1),
    steps = 32,
    cond_scale = 1.3
)
elapsed_time = (datetime.datetime.now() - start_date).total_seconds()

infilled_wave = rearrange(infilled_wave, "1 1 n -> 1 n")

# crop the conditioning wave from the output

infilled_wave = infilled_wave[:, cond_wave.shape[1]:]
infilled_duration = float(infilled_wave.shape[1]) / 24000
infilled_realtime_mult = infilled_duration / elapsed_time

print(f"Generated sample of duration {output_duration:0.2f}s in {elapsed_time}s ({infilled_realtime_mult:0.2f}x realtime)")

torchaudio.save(f"{args.tgt.replace('.wav','')}_condition.wav",infilled_wave.cpu(),24000)