import kagglehub
import os
import shutil

# Scarica il dataset
dataset_path = kagglehub.dataset_download(
    "kylegraupe/spy-daily-eod-options-quotes-2020-2022"
)

print("Dataset scaricato in:", dataset_path)

# Nome del file nel dataset
source_file = os.path.join(dataset_path, "spy_2020_2022.csv")

# Cartella dove stai eseguendo lo script
destination_file = os.path.join(os.getcwd(), "spy_2020_2022.csv")

# Copia il file nella cartella corrente
shutil.copy(source_file, destination_file)

print("CSV salvato qui:", destination_file)