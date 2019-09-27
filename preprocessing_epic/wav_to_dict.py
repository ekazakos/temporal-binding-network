import os
import argparse
import pickle
import multiprocessing as mp
import librosa


def read_and_resample(root, file):

    samples, sample_rate = librosa.core.load(os.path.join(root, file),
                                             sr=None,
                                             mono=False)

    print(sample_rate)
    return samples, file.split('.')[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sound_dir', help='Directory of EPIC audio')
    parser.add_argument('output_dir', help='Directory to save the pickled sound dictionary')
    parser.add_argument('--processes', type=int, default=40, help='Nummber of processes for multiprocessing')

    args = parser.parse_args()

    sound_dict = {}
    process_list = []
    pool = mp.Pool(processes=args.processes)
    for f in os.listdir(args.sound_dir):
        if f.endswith('.wav'):
            p = pool.apply_async(read_and_resample, (args.sound_dir, f))
            process_list.append(p)

    for p in process_list:
        samples, video_name = p.get()
        print(video_name)
        sound_dict[video_name] = samples

    pickle.dump(sound_dict, open(args.output_dir, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
