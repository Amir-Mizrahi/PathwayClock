import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
import argparse

class PathwayClockTrainer:
    def __init__(self, pathway_activity_file='moran_pathway_activity_matrix.txt',
                 labels_file='moran_ages.txt',
                 out_model_name='LongHack_v0.0.1.json'):
        self.out_model_name = out_model_name
        df = pd.read_csv(pathway_activity_file, sep=' ', header=None, index_col=0).T
        df.reset_index(inplace=True, drop=True)
        df['age'] = pd.read_csv(labels_file, header=None)
        train, test = train_test_split(df, test_size=0.2)
        self.X_train, self.y_train = train.iloc[:, :-1], train.iloc[:, -1]
        self.X_valid, self.y_valid = test.iloc[:, :-1], test.iloc[:, -1]

    def train(self):
        Xd = xgboost.DMatrix(self.X_train, label=self.y_train)
        self.model = xgboost.train({
            'eta': 1, 'max_depth': 6, 'base_score': 0, "lambda": 0
        }, Xd, 1)

        print("Model error =", np.linalg.norm(self.y_train - self.model.predict(Xd)))
        print(self.model.get_dump(with_stats=True)[0])

        self.model.save_model(self.out_model_name)

    def validate(self):
        Xdt = xgboost.DMatrix(self.X_valid)
        pred = self.model.predict(Xdt)
        print("validation r coeff: " + str(np.corrcoef(self.y_valid, pred)[0,1]))

    def run(self):
        self.train()
        self.validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', type=str)
    parser.add_argument('-l', '--label_file', type=str)
    parser.add_argument('-n', '--out_model_name', type=str)
    args = parser.parse_args()

    trainer = PathwayClockTrainer(args.data_file, args.label_file, args.out_model_name)
    trainer.run()