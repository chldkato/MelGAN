import numpy as np
import argparse, os, librosa, torch, glob, shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm


def main(args):
    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    make_testset()
    trainset = glob.glob(os.path.join(args.data_dir, '*.wav'))
    trainset = trainset[args.valid_num + args.test_num:]
    for train_path in tqdm(trainset):
        train_name = train_path.split('\\')[-1]
        executor.submit(process_audio(train_path, train_name))
    print('Trainset Done')

    validset = glob.glob(os.path.join(args.valid_dir, '*.wav'))
    for valid_path in tqdm(validset):
        process_audio(valid_path)
    print('Validset Done')


def process_audio(wav_path, wav_name=None):
    wav, sr = librosa.core.load(wav_path, sr=22050)
    mel_basis = librosa.filters.mel(sr, 1024, 80)
    spectrogram = librosa.stft(y=wav, n_fft=1024, hop_length=256, win_length=1024)
    mel_spectrogram = np.dot(mel_basis, np.abs(spectrogram)).astype(np.float32)
    mel_spectrogram = torch.from_numpy(mel_spectrogram)

    if wav_name is not None:
        wav_path = os.path.join(args.train_dir, wav_name)
    save_path = wav_path.replace('.wav', '.mel')
    torch.save(mel_spectrogram, save_path)
    librosa.output.write_wav(wav_path, wav, sr)


def make_testset():
    for idx, fn in enumerate(os.listdir(args.data_dir)):
        if idx + 1 <= args.valid_num:
            shutil.copy(os.path.join(args.data_dir, fn), os.path.join(args.valid_dir))
            continue
        shutil.copy(os.path.join(args.data_dir, fn), os.path.join(args.test_dir))
        if idx + 1 == args.test_num + args.valid_num:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/melgan'))
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--train_dir', default='./train')
    parser.add_argument('--valid_dir', default='./valid')
    parser.add_argument('--test_dir', default='./test')
    parser.add_argument('--valid_num', default=2)
    parser.add_argument('--test_num', default=2)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    train_dir = os.path.join(args.train_dir)
    os.makedirs(train_dir, exist_ok=True)
    valid_dir = os.path.join(args.valid_dir)
    os.makedirs(valid_dir, exist_ok=True)
    test_dir = os.path.join(args.test_dir)
    os.makedirs(test_dir, exist_ok=True)
    main(args)
