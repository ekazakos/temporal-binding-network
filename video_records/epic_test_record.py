from .video_record import VideoRecord


class EpicTestVideoRecord(VideoRecord):
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
    def num_frames(self):
        return {'RGB': self.end_frame-self.start_frame, 'Flow': (self.end_frame-self.start_frame)/2,
                'Spec': self.end_frame-self.start_frame}
