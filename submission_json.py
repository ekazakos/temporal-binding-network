import json
from pathlib import Path
import numpy as np
import pandas as pd

from epic_kitchens.meta import training_labels


def softmax(x):
    '''
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
    '''
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def top_scores(scores):
    top_n_scores_idx = np.argsort(scores)[:, ::-1]
    top_n_scores = scores[np.arange(0, len(scores)).reshape(-1, 1), top_n_scores_idx]
    return top_n_scores_idx, top_n_scores


def compute_action_scores(verb_scores, noun_scores, n=100):
    top_verbs, top_verb_scores = top_scores(verb_scores)
    top_nouns, top_noun_scores = top_scores(noun_scores)
    top_verb_probs = softmax(top_verb_scores)
    top_noun_probs = softmax(top_noun_scores)
    action_probs_matrix = top_verb_probs[:, :n, np.newaxis] * top_noun_probs[:, np.newaxis, :]
    instance_count = action_probs_matrix.shape[0]
    action_ranks = action_probs_matrix.reshape(instance_count, -1).argsort(axis=-1)[:, ::-1]
    verb_ranks_idx, noun_ranks_idx = np.unravel_index(action_ranks[:, :n],
                                                      dims=(action_probs_matrix.shape[1:]))

    # TODO: Reshape, argsort, then convert back to verb/noun indices
    segments = np.arange(0, instance_count).reshape(-1, 1)
    return ((top_verbs[segments, verb_ranks_idx], top_nouns[segments, noun_ranks_idx]),
            action_probs_matrix.reshape(instance_count, -1)[segments, action_ranks[:, :n]])


def action_scores_to_json(actions, scores, prior):
    entries = []
    for verbs, nouns, segment_scores in zip(*actions, scores):
        if prior is None:
            entries.append({"{},{}".format(verb, noun): float(score) for verb, noun, score in zip(verbs, nouns, segment_scores)})
        else:
            entries.append({"{},{}".format(verb, noun): (float(prior[(verb, noun)]) if (verb, noun) in prior else 0.0) * float(score) for verb, noun, score in zip(verbs, nouns, segment_scores)})
    return entries


def scores_to_json(scores):
    entries = []
    for classes, segment_scores in zip(*top_scores(scores)):
        entries.append({str(cls): float(score) for cls, score in zip(classes, segment_scores)})
    return entries


def compute_score_dicts(results, test_set, prior):
    verb_scores = results['test_' + test_set + '_scores']['verb']
    if len(verb_scores.shape) == 4:
        verb_scores = verb_scores.mean(axis=(1, 2))
    noun_scores = results['test_' + test_set + '_scores']['noun']
    if len(noun_scores.shape) == 4:
        noun_scores = noun_scores.mean(axis=(1, 2))
    actions, action_scores = compute_action_scores(verb_scores, noun_scores)

    verb_scores_dict = scores_to_json(verb_scores)
    noun_scores_dict = scores_to_json(noun_scores)
    action_scores_dict = action_scores_to_json(actions, action_scores, prior)
    return verb_scores_dict, noun_scores_dict, action_scores_dict


def to_json(uids, verb_scores_dict, noun_scores_dict, action_scores_dict):
    entries = {}
    for uid, segment_verb_scores_dict, segment_noun_scores_dict, segment_action_scores_dict in zip(uids,
                                                                                                   verb_scores_dict,
                                                                                                   noun_scores_dict,
                                                                                                   action_scores_dict):
        entries[str(uid)] = {
            'verb': segment_verb_scores_dict,
            'noun': segment_noun_scores_dict,
            'action': segment_action_scores_dict
        }

    return {
        'version': '0.1',
        'challenge': 'action_recognition',
        'results': entries,
    }


def dump_scores_to_json(results, uids, filepath, test_set, prior):
    verb_scores_dict, noun_scores_dict, action_scores_dict = compute_score_dicts(results, test_set, prior)
    results_dict = to_json(uids, verb_scores_dict, noun_scores_dict, action_scores_dict)

    filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(results_dict, f)
    return results_dict


def main(args):

    if not args.submission_json.exists():
        args.submission_json.mkdir(parents=True, exist_ok=True)
    for test_set in ['seen', 'unseen']:
        if test_set == 'unseen':
            action_counts = training_labels().apply(lambda d: (d['verb_class'], d['noun_class']), axis=1).value_counts()
            prior_action = action_counts.div(action_counts.sum())
            prior = prior_action
        else:
            prior = None
        results = pd.read_pickle(args.results_dir / ('test_' + test_set + '.pkl'))
        uids = np.zeros(results['test_' + test_set + '_scores']['verb'].shape[0], dtype=np.int)
        ts = 's1' if test_set == 'seen' else 's2'
        timestamps = pd.read_pickle(args.annotations_dir / ('EPIC_test_' + ts + '_timestamps.pkl'))
        for i, (idx, row) in enumerate(timestamps.iterrows()):
            uids[i] = str(idx)
        dump_scores_to_json(results, uids, args.submission_json / (test_set + '.json'), test_set, prior)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce submission JSON from results pickle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("annotations_dir", type=Path)
    parser.add_argument("submission_json", type=Path)
    main(parser.parse_args())