# MelGAN

Paper: [MelGAN](https://arxiv.org/abs/1910.06711)

Official repository
  * https://github.com/descriptinc/melgan-neurips

logs 폴더에 1200 epoch까지 학습한 모델이 있습니다

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

2. **`~/data`에 학습 데이터 준비**

   ```
   data
     |- 1_0000.wav
     |- ...
     |- 4_5631.wav
   ```
     * 사용할 wav 파일만 있으면 됩니다

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * train, valid, test 폴더가 생성되고 해당되는 wav, mel파일이 위치하게 됩니다

4. **Train**
   ```
   python train.py
   ```

   재학습 시
   ```
   python train.py --load_dir logs/ckpt-970k.pt
   ```
     * 숫자를 변경하면 됩니다

5. **Inference**
   ```
   python eval.py --load_dir logs/ckpt-970k.pt
   ```
     * 숫자를 변경하면 됩니다. output 폴더에 wav파일과 스펙트로그램 png 파일이 생성됩니다

#### Spectrogram example
![spectrogram-0](https://user-images.githubusercontent.com/49984198/78754810-1bc19600-79b3-11ea-9240-e421a1821cdb.png)