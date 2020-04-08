import numpy as np
import os, argparse, glob, librosa, librosa.display, torch, scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
from gan import Generator
from pathlib import Path
from matplotlib import cm


def main(args):
    vocoder = Generator(80)
    vocoder = vocoder.cuda()
    ckpt = torch.load(args.load_dir)
    vocoder.load_state_dict(ckpt['G'])
    testset = glob.glob(os.path.join(args.test_dir, '*.wav'))
    for i, test_path in enumerate(tqdm(testset)):
        mel, spectrogram = process_audio(test_path)
        g_audio = vocoder(mel.cuda())
        g_audio = g_audio.squeeze().cpu()
        audio = (g_audio.detach().numpy() * 32768)
        g_spec = librosa.stft(y=audio, n_fft=1024, hop_length=256, win_length=1024)
        scipy.io.wavfile.write(Path(args.save_dir) / ('generated-%d.wav' % i), 22050, audio.astype('int16'))
        plot_stft(spectrogram, g_spec, i)


def process_audio(wav_path):
    wav, sr = librosa.core.load(wav_path, sr=22050)
    mel_basis = librosa.filters.mel(sr, 1024, 80)
    spectrogram = librosa.stft(y=wav, n_fft=1024, hop_length=256, win_length=1024)
    mel_spectrogram = np.dot(mel_basis, np.abs(spectrogram)).astype(np.float32)
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    save_path = wav_path.replace('.wav', '.mel')
    torch.save(mel_spectrogram, save_path)
    return mel_spectrogram.unsqueeze(0), spectrogram


def plot_stft(spectrogram, g_spec, idx):
    plt.figure(figsize=(12, 8))

    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', hop_length=256)
    plt.title('original audio spectrogram')

    g_spec = librosa.amplitude_to_db(np.abs(g_spec), ref=np.max)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(g_spec, x_axis='time', y_axis='log', hop_length=256)
    plt.title('generated audio spectrogram')

    plt.tight_layout()
    fn = 'spectrogram-%d.png' % idx
    plt.savefig(args.save_dir + '/' + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='./test')
    parser.add_argument('--load_dir', required=True)
    parser.add_argument('--save_dir', default='./output')
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    main(args)
