from video_records import EpicVideoRecord
import torch.utils.data as data

import librosa
from PIL import Image
import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import randint
import pickle


class TBNDataSet(data.Dataset):
    def __init__(self, dataset, list_file,
                 new_length, modality, image_tmpl,
                 visual_path=None, audio_path=None,
                 fps=29.94, resampling_rate=44000,
                 num_segments=3, transform=None,
                 mode='train'):
        self.dataset = dataset
        if audio_path is not None:
            if self.dataset != 'epic':
                self.audio_path = Path(audio_path)
            else:
                self.audio_path = pickle.load(open(audio_path, 'rb'))
        self.visual_path = visual_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.mode = mode
        self.resampling_rate = resampling_rate
        self.fps = fps

        if 'RGBDiff' in self.modality:
            self.new_length['RGBDiff'] += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _log_specgram(self, audio, window_size=10,
                     step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.resampling_rate / 1e3))
        noverlap = int(round(step_size * self.resampling_rate / 1e3))

        spec = librosa.stft(audio, n_fft=511,
                            window='hann',
                            hop_length=noverlap,
                            win_length=nperseg,
                            pad_mode='constant')

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _extract_sound_feature(self, record, idx):

        centre_sec = (record.start_frame + idx) / self.fps
        left_sec = centre_sec - 0.639
        right_sec = centre_sec + 0.639
        audio_fname = record.untrimmed_video_name + '.wav'
        if self.dataset != 'epic':
            samples, sr = librosa.core.load(self.audio_path / audio_fname,
                                            sr=None, mono=True)

        else:
            audio_fname = record.untrimmed_video_name
            samples = self.audio_path[audio_fname]

        duration = samples.shape[0] / float(self.resampling_rate)

        left_sample = int(round(left_sec * self.resampling_rate))
        right_sample = int(round(right_sec * self.resampling_rate))

        if left_sec < 0:
            samples = samples[:int(round(self.resampling_rate * 1.279))]

        elif right_sec > duration:
            samples = samples[-int(round(self.resampling_rate * 1.279)):]
        else:
            samples = samples[left_sample:right_sample]

        return self._log_specgram(samples)

    def _load_data(self, modality, record, idx):
        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame + idx
            return [Image.open(os.path.join(self.visual_path, record.untrimmed_video_name, self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')]
        elif modality == 'Flow':
            idx_untrimmed = int(np.floor((record.start_frame / 2))) + idx
            # idx_untrimmed = record.start_frame + idx
            x_img = Image.open(os.path.join(self.visual_path, record.untrimmed_video_name, self.image_tmpl[modality].format('x', idx_untrimmed))).convert('L')
            y_img = Image.open(os.path.join(self.visual_path, record.untrimmed_video_name, self.image_tmpl[modality].format('y', idx_untrimmed))).convert('L')

            return [x_img, y_img]
        elif modality == 'Spec':
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]

    def _parse_list(self):
        if self.dataset == 'epic':
            self.video_list = [EpicVideoRecord(tup) for tup in pd.read_pickle(self.list_file).iterrows()]


    def _sample_indices(self, record, modality):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        # elif record.num_frames[modality] > self.num_segments:
        #     offsets = np.sort(randint(record.num_frames[modality] - self.new_length[modality] + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record, modality):

        tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):

        input = {}
        record = self.video_list[index]

        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # We implement a Temporal Binding Window (TBW) with size same as the action's length by:
            #   1. Selecting different random indices (timestamps) for each modality within segments
            #      (this is similar to using a TBW with size same as the segment's size)
            #   2. Shuffling randomly the segments of Flow, Audio (RGB is the anchor hence not shuffled)
            #      which binds data across segments, hence making the TBW same in size as the action.
            #   Example of an action with 90 frames across all modalities:
            #    1. Synchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [12, 41, 80], Audio: [12, 41, 80]
            #    2. Asynchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [9, 55, 88], Audio: [20, 33, 67]
            #    3. Asynchronous selection of indices per action:
            #       RGB: [12, 41, 80], Flow: [88, 55, 9], Audio: [67, 20, 33]

            if m != 'RGB' and self.mode == 'train':
                np.random.shuffle(segment_indices)

            img, label = self.get(m, record, segment_indices)
            input[m] = img

        return input, label

    def get(self, modality, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1

        process_data = self.transform[modality](images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
