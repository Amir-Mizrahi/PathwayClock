from training import PathwayClockTrainer
import os
import warnings
warnings.filterwarnings("ignore")

prefixes = ["human_brain_transcriptome_atlas", "moran"]

summary = []
for prefix in prefixes:
    label_file = prefix + "_ages.txt"
    for file in os.listdir('data'):
        if file.startswith(prefix + "_pathway"):
            experiment = os.path.splitext(file)[0]
            #print(experiment)
            trainer = PathwayClockTrainer(os.path.join("data",file), os.path.join("data",label_file),
                                          os.path.join("models",experiment + "_model.json"))
            results = trainer.run()
            #print("corr coeff: %.2f (%.2f)" % (results.mean(), results.std()))
            summary.append(experiment + " , corr coeff mean: %.2f ,std: %.2f" % (results.mean(), results.std()))

with open('summary.txt','w') as f:
    f.writelines('\n'.join(summary))
