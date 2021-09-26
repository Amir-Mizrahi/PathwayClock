from pathway_clock import *
from training import *

data_file = "moran_pathway_kegg_sd_activity_matrix_model.json"
label_file = "moran_ages.txt"
clinical_outcomes = "moran_clinical_outcome.txt"

class PathwayClockTrainer:
    def __init__(self, pathway_activity_file='data\moran_pathway_activity_matrix.txt',
                 labels_file='data\moran_ages.txt',
                 clinical_outcome_file='data\moran_clinical_outcome.txt',
                 out_model_name='models\LongHack_v0.0.1.json'):
        self.out_model_name = out_model_name
        df = pd.read_csv(pathway_activity_file, sep=' ', header=None, index_col=0).T
        df.reset_index(inplace=True, drop=True)
        df['age'] = pd.read_csv(labels_file, header=None)
        df['co'] = pd.read_csv(clinical_outcome_file, header=None)
        train = df[df['co'] == 'control']
        train = train.drop(columns=['co'])
        test = df[df['co'] == 'parkinson']
        test = test.drop(columns=['co'])
        self.X_train, self.y_train = train.iloc[:, :-1], train.iloc[:, -1]
        self.X_test, self.y_test = test.iloc[:, :-1], test.iloc[:, -1]

    def train(self):
        Xd = xgboost.DMatrix(self.X_train, label=self.y_train)
        params = {
            'eta': 1, 'max_depth': 3, 'base_score': 0, "lambda": 0
        }
        self.model = xgboost.train(params, Xd, 1)
        #print("Model error =", np.linalg.norm(self.y_train - self.model.predict(Xd)))
        #print(self.model.get_dump(with_stats=True)[0])
        self.model.save_model(self.out_model_name)
        #return results

    def validate(self):
        Xdt = xgboost.DMatrix(self.X_test)
        pred = self.model.predict(Xdt)
        print("validation r coeff: " + str(np.corrcoef(self.y_test, pred)[0,1]))

    def run(self):
        self.train()
        self.validate()

if __name__ == '__main__':
    clinical_outcome_file = 'data\moran_clinical_outcome.txt'
    data_file = 'data\moran_pathway_kegg_sd_activity_matrix.txt'
    model_file = 'models\moran_pathway_kegg_sd_activity_control.json'
    labels_file = 'data\moran_ages.txt'
    trainer = PathwayClockTrainer(pathway_activity_file=data_file,
                                  out_model_name=model_file,
                                  labels_file=labels_file,
                                  clinical_outcome_file=clinical_outcome_file)
    #results = trainer.run()
   # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    df = pd.read_csv(data_file, sep=' ', header=None, index_col=0).T
    df.reset_index(inplace=True, drop=True)
    df['co'] = pd.read_csv(clinical_outcome_file, header=None)
    test = df[df['co'] == 'parkinson']
    test = test.drop(columns=['co'])
    pathway_clock = PathwayClock(model_path=trainer.out_model_name)
    resutls_df = pd.DataFrame(pathway_clock.predict(test))
    df['age'] = pd.read_csv(labels_file, header=None)
    resutls_df['age']=list(df[df['co'] == 'parkinson']['age'])
    resutls_df = resutls_df[['age','predicted_age',0,1,2]]
    #resutls_df.to_csv("pd_patients_predictions.csv")
    print(np.corrcoef(resutls_df['age'], resutls_df['predicted_age']))
