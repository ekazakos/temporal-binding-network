from pathlib import Path
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path, help='Directory of epic dataset')
    parser.add_argument('symlinks_dir', type=Path, help='Directory to save symlinks for EPIC')

    args = parser.parse_args()

    if not args.symlinks_dir.exists():
        args.symlinks_dir.mkdir(parents=True)

    for modality in ['rgb', 'flow']:
        if modality == 'rgb':
            pattern = 'P[0-3][0-9]/P[0-3][0-9]_[0-9][0-9]/'
        else:
            pattern = 'P[0-3][0-9]/P[0-3][0-9]_[0-9][0-9]/*/'
        for split in ['train', 'test']:
            modality_split_dir = args.data_dir / modality / split
            for source_file in modality_split_dir.glob(pattern):
                if modality == 'rgb':
                    person, video = str(source_file).split('/')[-2:]
                else:
                    person, video, _ = str(source_file).split('/')[-3:]

                link_path = args.symlinks_dir / video
                if not link_path.exists():
                    link_path.mkdir(parents=True)

                for i, _ in enumerate(source_file.iterdir()):

                    f = 'frame_{:010d}.jpg'.format(i + 1)
                    source = source_file / f

                    if modality == 'rgb':
                        link = link_path / 'img_{:010d}.jpg'.format(i)
                    else:
                        if source_file.name == 'u':
                            link = link_path / 'x_{:010d}.jpg'.format(i)

                        else:
                            link = link_path / 'y_{:010d}.jpg'.format(i)

                    if link.exists():
                        link.unlink()
                    link.symlink_to(source)
