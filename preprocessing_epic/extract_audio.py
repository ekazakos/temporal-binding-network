import argparse
import subprocess
import os


def ffmpeg_extraction(input_video, output_sound, sample_rate):

    ffmpeg_command = ['ffmpeg', '-i', input_video,
                      '-vn', '-acodec', 'pcm_s16le',
                      '-ac', '1', '-ar', sample_rate,
                      output_sound]

    subprocess.call(ffmpeg_command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Directory of EPIC videos with audio')
    parser.add_argument('output_dir', help='Directory of EPIC videos with audio')
    parser.add_argument('--sample_rate', default='24000', help='Rate to resample audio')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for root, dirs, files in os.walk(args.videos_dir):
        for f in files:
            if f.endswith('.MP4'):
                ffmpeg_extraction(os.path.join(root, f),
                                  os.path.join(args.output_dir,
                                               os.path.splitext(f)[0] + '.wav'),
                                  args.sample_rate)
