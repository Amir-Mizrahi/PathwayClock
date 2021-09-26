import numpy as np
import pandas as pd
import xgboost
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

class PathwayClockTrainer:
    def __init__(self, pathway_activity_file='moran_pathway_activity_matrix.txt',
                 labels_file='moran_ages.txt',
                 out_model_name='LongHack_v0.0.1.json'):
        self.out_model_name = out_model_name
        df = pd.read_csv(pathway_activity_file, sep=' ', header=None, index_col=0).T
        df.reset_index(inplace=True, drop=True)
        df['age'] = pd.read_csv(labels_file, header=None)
        self.X_train, self.y_train = df.iloc[:, :-1], df.iloc[:, -1]

    def train(self):
        Xd = xgboost.DMatrix(self.X_train, label=self.y_train)
        params = {
            'eta': 1, 'max_depth': 3, 'base_score': 0, "lambda": 0
        }
        classifier = xgboost.XGBClassifier(params, verbosity=0)
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        results = cross_val_score(classifier, self.X_train, self.y_train, cv=kfold,
                                  scoring=make_scorer(lambda x,y: np.corrcoef(x,y)[0,1]))
        #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        self.model = xgboost.train(params, Xd, 1)
        #print("Model error =", np.linalg.norm(self.y_train - self.model.predict(Xd)))
        #print(self.model.get_dump(with_stats=True)[0])
        self.model.save_model(self.out_model_name)
        return results

    def run(self):
        return self.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', type=str)
    parser.add_argument('-l', '--label_file', type=str)
    parser.add_argument('-n', '--out_model_name', type=str)
    args = parser.parse_args()

    trainer = PathwayClockTrainer(args.data_file, args.label_file, args.out_model_name)
    results = trainer.run()
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))