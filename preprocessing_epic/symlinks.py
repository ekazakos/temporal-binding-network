from pathlib import Path
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path, help='Directory of epic dataset')
    parser.add_argument('symlinks_dir', type=Path, help='Directory to save symlinks for EPIC')

    args = parser.parse_args()

    if not args.symlinks_dir.exists():
        args.symlinks_dir.mkdir(parents=True)

    data_dir = args.data_dir
    participant_pattern = 'P??'

    for participant_dir in data_dir.glob(participant_pattern):
        for modality in ['rgb_frames', 'flow_frames']:
            if modality == 'rgb_frames':
                video_id_pattern = 'P??_*??/'
            else:
                video_id_pattern = 'P??_*??/*/'

            frames_dir = participant_dir / modality
            for source_file in frames_dir.glob(video_id_pattern):
                if modality == 'rgb_frames':
                    video = str(source_file).split('/')[-1:]
                else:
                    video, _ = str(source_file).split('/')[-2:]

                link_path = args.symlinks_dir / video
                if not link_path.exists():
                    link_path.mkdir(parents=True)

                for i, _ in enumerate(source_file.iterdir()):

                    f = 'frame_{:010d}.jpg'.format(i + 1)
                    source = source_file / f

                    if modality == 'rgb_frames':
                        link = link_path / 'img_{:010d}.jpg'.format(i)
                    else:
                        if source_file.name == 'u':
                            link = link_path / 'x_{:010d}.jpg'.format(i)

                        else:
                            link = link_path / 'y_{:010d}.jpg'.format(i)

                    if link.exists():
                        link.unlink()
                    link.symlink_to(source)
