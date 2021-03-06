import numpy as np
import json
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 18})

dataset_names = [
    ('arcene', 7000, 40., 80.),
    ('dexter', 9947, 70., 94.),
    ('gisette', 2500, 90., 100.),
    ('madelon', 20, 50., 93.),
    ('dorothea', 50000, 60., 95.)
]

if __name__ == '__main__':

    root_directory = './info/'


    for dataset_stats in dataset_names:
        dataset_name, max_features, ylow, yhigh = dataset_stats
        print('DATASET : ', dataset_name)
        plt.ion()
        plt.figure()
        legends = []

        output_filename = root_directory + dataset_name + '_results.json'
        try:
            with open(output_filename) as outfile:
                results = json.load(outfile)
        except:
            continue

        for method in results:
            rank_list = results[method]
            if method == 'lasso':
                continue
            for example in rank_list:
                print('METHOD : ', method)
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

                best_acc_index = np.argmax(accuracy)
                print('Best acc : ', accuracy[best_acc_index], ', nfeatures : ', nfeatures[best_acc_index])

                # print('Mean acc : ',  accuracy.mean(), ', std : ', accuracy.std())

                legend_name = '$\gamma = ' + ('%.2f' % example['gamma']) + '$'
                legends.append(legend_name)

                nfeatures_diff = nfeatures[1:] - nfeatures[:-1]
                accuracy_mean = 0.5 * (accuracy[1:] + accuracy[:-1])

                roc = np.sum(accuracy_mean * nfeatures_diff)
                roc /= (nfeatures[-1] - nfeatures[0])
                print('ROC : ', roc)

                # plt.ion()
                positions = np.where(nfeatures < 1000)[0]
                plt.plot(
                    nfeatures[positions], accuracy[positions]
                )

                if 'percentiles' in example:
                    for per in example['percentiles']:
                        print(per)

        plt.title(dataset_name)
        plt.legend(tuple(legends), prop={'size': 12}, loc='best')
        plt.ylim(ylow, yhigh)
        plt.show()
        plt.xlabel('# features')
        plt.ylabel('accuracy')
        plt.tight_layout()
        plt.savefig(root_directory + dataset_name + '_gamma.png')
