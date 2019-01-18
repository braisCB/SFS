import numpy as np
import json
from scipy.stats import friedmanchisquare, wilcoxon


dataset_names = [
    ('arcene', 7000),
    ('dexter', 9947),
    ('gisette', 2500),
    ('madelon', 20),
    ('dorothea', 50000)
]
gamma = 0.0

if __name__ == '__main__':

    root_directory = './info/'

    global_ranking = []

    for dataset_stats in dataset_names:
        dataset_name, max_features = dataset_stats
        print('DATASET : ', dataset_name)
        legends = []

        output_filename = root_directory + dataset_name + '_gamma_' + str(gamma) + '_results.json'
        try:
            with open(output_filename) as outfile:
                results = json.load(outfile)
        except:
            continue

        methods = []
        accuracies = []
        all_accuracies = []
        for method in results:
            rank_list = results[method]
            if method == 'lasso':
                continue
            for example in rank_list:
                if 'lasso' in example:
                    print('lasso : ', example['lasso'])
                if 'gamma' in example:
                    print('gamma : ', example['gamma'])
                if 'reps' in example:
                    print('reps : ', example['reps'])
                accuracy = np.array(example['classification']['accuracy'])
                nfeatures = np.array(example['classification']['n_features'])

                positions = np.where(nfeatures <= max_features)[0]
                nfeatures = nfeatures[positions]
                accuracy = np.mean(accuracy[positions], axis=-1)
                all_accuracies.append(accuracy)
                accuracies.append(accuracy)
                methods.append(method + '_' + str(example['reps']))

        accuracies = np.array(accuracies).T

        ranking = np.zeros(accuracies.shape[1])
        for accuracy in accuracies:
            unique = np.sort(np.unique(accuracy))[::-1]
            for i, acc in enumerate(accuracy):
                ranking[i] += np.where(unique == acc)[0] + 1.
        ranking /= accuracies.shape[0]

        global_ranking.append(ranking)

        for method, ranking in zip(methods, ranking):
            print('METHOD : ', method, ' , RANKING : ', ranking)

        val, p = friedmanchisquare(all_accuracies[0], all_accuracies[1], all_accuracies[2], all_accuracies[3], all_accuracies[4])
        val2, p2 = friedmanchisquare(all_accuracies[1], all_accuracies[2], all_accuracies[3])
        val3, p3 = friedmanchisquare(all_accuracies[1], all_accuracies[2], all_accuracies[3], all_accuracies[4])
        val4, p4 = wilcoxon(all_accuracies[0], all_accuracies[2])
        val5, p5 = wilcoxon(all_accuracies[0], all_accuracies[3])
        val5, p6 = wilcoxon(all_accuracies[0], all_accuracies[4])
        print('FRIEDMAN 1,2,3,4,5 : ', p)
        print('FRIEDMAN 2,3,4 : ', p2)
        print('FRIEDMAN 2,3,4,5 : ', p3)
        print('WILCOXON 1,3 : ', p4)
        print('WILCOXON 1,4 : ', p5)
        print('WILCOXON 1,5 : ', p6)
