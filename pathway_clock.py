import xgboost
import shap
import pandas as pd
import numpy as np
import argparse

class PathwayClock:
    def __init__(self, model_path="models\LongHack_v0.0.1.json"):
        self.model = xgboost.Booster()
        self.model.load_model(model_path)
        self.explainer = shap.Explainer(self.model)

    def predict(self, X_test):
        Xdt = xgboost.DMatrix(X_test)
        pred = self.model.predict(Xdt)
        shap_values = self.explainer(X_test)
        df_results = pd.DataFrame()
        df_results["predicted_age"] = pred
        top_pathways = np.apply_along_axis(lambda x: x.argsort()[-3:][::-1], 1, shap_values.values)
        df_results = pd.concat([df_results, pd.DataFrame(X_test.columns[top_pathways], index=df_results.index)], axis=1)
        return df_results.to_dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', type=str)
    parser.add_argument('-m', '--model_file', type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.data_file, sep=' ', header=None, index_col=0).T
    df.reset_index(inplace=True, drop=True)
    if args.model_file is not None:
        pathway_clock = PathwayClock(args.model_file)
    else:
        pathway_clock = PathwayClock()

    print(pathway_clock.predict(df))
