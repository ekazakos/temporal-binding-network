from pathlib import Path
import argparse
import pickle

import pandas as pd
import numpy as np


def softmax(x):
    """
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    """
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def fuse_scores(scores_dict, split):
    modalities_combinations = [('rgb', 'flow'), ('rgb', 'spec'),
                               ('flow', 'spec'), ('rgb', 'flow', 'spec')]

    fused_scores = {}
    for mod_comb in modalities_combinations:
        name = '_'.join(mod_comb)
        fused_scores[name] = {'test_' + split + '_scores': {}}
        for task in ['verb', 'noun']:
            scores_list = [scores_dict[m]['test_' + split + '_scores'][task] for m in mod_comb]
            scores_list = [softmax(scores.mean(axis=(1, 2))) for scores in scores_list]
            fused_scores[name]['test_' + split + '_scores'][task] = np.mean(scores_list, axis=0)

    return fused_scores


def main(args):

    for split in ['seen', 'unseen']:
        rgb_scores = pd.read_pickle(args.rgb / ('test_' + split + '.pkl'))
        flow_scores = pd.read_pickle(args.flow / ('test_' + split + '.pkl'))
        spec_scores = pd.read_pickle(args.spec / ('test_' + split + '.pkl'))

        scores_dict = {'rgb': rgb_scores, 'flow': flow_scores, 'spec': spec_scores}
        fused_scores = fuse_scores(scores_dict, split)

        for key in fused_scores.keys():
            output_dir = args.scores_root / key / ('test_' + split + '.pkl')
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(output_dir, 'wb') as f:
                pickle.dump(fused_scores[key], f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('scores_root', type=Path)
    parser.add_argument('--rgb', type=Path, help='Directory of the RGB scores')
    parser.add_argument('--flow', type=Path, help='Directory of the Flow scores')
    parser.add_argument('--spec', type=Path, help='Directory of the Spectrogram scores')

    args = parser.parse_args()
    main(args)
