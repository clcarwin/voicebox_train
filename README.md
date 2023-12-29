# voicebox with full train code
1. The code modified base on ```voicebox-pytorch 0.4.11```
2. This code only train S2S, not for TTS
3. [pull 39](https://github.com/lucidrains/voicebox-pytorch/pull/39) is very important

# Prepare
1. Download vocos-encodec-24khz
```bash
mkdir checkpoints
cd checkpoints
git clone https://huggingface.co/charactr/vocos-encodec-24khz
```

2. Download hubert and km2000 to ```checkpoints``` directory

[Expresso Page](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso)<br>
[hubert_base_ls960.pt](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)<br>
[hubert_base_ls960_L9_km2000_expresso.bin](https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hubert_base_ls960_L9_km2000_expresso.bin)

3. Pretrained model

[Discussion Page](https://github.com/lucidrains/voicebox-pytorch/discussions/29#discussioncomment-7732769)<br>
[voicebox_small.pt](https://huggingface.co/lucasnewman/voicebox-small)

4. Dowanload LibriTTS-R dataset

# Inference
```bash
python demo.py --tgt assets/1841_r960_3.wav --ref assets/5717_r960_0.wav --ckpt checkpoints/voicebox_small.pt
# output: assets/1841_r960_3_uncondition.wav
# output: assets/1841_r960_3_condition.wav
```

# Training
```bash
python train.py --train /work/data/LibriTTS_R --logs ./logs
```
