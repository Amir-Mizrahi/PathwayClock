from pathway_clock import *
from training import *


if __name__ == '__main__':
    data_file = 'data\human_brain_transcriptome_atlas_pathway_kegg_sd_activity_matrix.txt'
    model_file = 'models\human_brain_transcriptome_atlas_pathway_kegg_sd_activity_matrix_model.json'
    labels_file = 'data\human_brain_transcriptome_atlas_ages.txt'
    trainer = PathwayClockTrainer(pathway_activity_file=data_file,
                                  out_model_name=model_file,
                                  labels_file=labels_file)
    #results = trainer.run()
   # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    df = pd.read_csv(data_file, sep=' ', header=None, index_col=0).T
    df.reset_index(inplace=True, drop=True)

    pathway_clock = PathwayClock(model_path=trainer.out_model_name)
    resutls_df = pd.DataFrame(pathway_clock.predict(df))
    resutls_df['age'] = pd.read_csv(labels_file, header=None)
    resutls_df = resutls_df[['age','predicted_age',0,1,2]]
    resutls_df.to_csv("brain_atlas_patients_predictions.csv")
    print(np.corrcoef(resutls_df['age'], resutls_df['predicted_age']))