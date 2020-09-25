from .video_record import VideoRecord


class EpicKitchens100_VideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def fps(self):
        is_2020 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return {'RGB': 50 if is_2020 else 60,
                'Flow': 50 if is_2020 else 30,
                'Spec': 50 if is_2020 else 60}

    @property
    def num_frames(self):
        rgb2flow_fps_ratio = self.fps['Flow'] / float(self.fps['RGB'])
        return {'RGB': self.end_frame - self.start_frame,
                'Flow': (self.end_frame - self.start_frame) * rgb2flow_fps_ratio,
                'Spec': self.end_frame - self.start_frame}

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -10000,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -10000}
