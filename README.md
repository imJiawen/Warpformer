# Warpformer

This is an offical implementation of [Warpformer: A Multi-scale Modeling Approach for Irregular Clinical Time Series](https://arxiv.org/abs/2306.09368).


## Setup

   ### Requirements

   Your local system should have the following executables:

   - [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   - Python 3.7 or later
   - git
   - [PostreSQL](http://www.postgresql.org/download/) (Opt.)

   ### Create conda environment

   All instructions below should be executed from a terminal.

   1. clone this repository and run 
   ```bash
   cd Warpformer
   ```
   2. create an environment ```warpformer``` and activate it.
   ```bash
   conda create -n warpformer python=3.7
   conda activate warpformer
   ```
   3. install the required Python modules using file [requirements.txt](requirements.txt).
   ```bash
   pip install -r requirement.txt
   ```
   Tips: If you cannot install psycopg2 successfully, please try ```sudo apt-get install libpq-dev``` first and then run the command ```pip install psycopg2```.


## Download Clinical Data & Task Building

   Our benchmark includes a diverse set of clinical tasks covering different clinical scenarios, with 61 common physiology signals and 42 widely used interventions in intensive care units. The following table summarizes the statistics of these tasks.

   |  Task (Abbr.)   | Type  | # Train | # Val. | # Test | Clinical Scenario |
   |  :----  | :----: | ----: | ----: | ----: | :---- |
   | In-hospital Mortality (MOR)             | BC | 39, 449    | 4, 939  | 4, 970 | Early warning |
   | Decompensation (DEC)                    | BC | 249, 045    | 31, 896 | 30, 220 | Outcome pred. |
   | Length Of Stay (LOS)                    | MC | 249, 572   | 31, 970 | 30, 283 | Outcome pred. |
   | Next Timepoint Will Be Measured (WBM)   | ML | 223, 867   | 28, 754 | 27, 038 | Treatment recom. |
   | Clinical Intervention Prediction (CIP)  | MC | 223, 913   | 28, 069 | 27, 285 | Treatment recom. |


   ### I. Access to MIMIC-III data

   1. First you need to have an access to MIMIC-III Dataset, which can be requested [here](https://mimic.physionet.org/gettingstarted/access/). The database version we used here is v1.4.
   2. Download the MIMIC-III Clinical Database and place the MIMIC-III Clinical Database as either .csv or .csv.gz files somewhere on your local computer.


   ### II. Generate datasets

   1. Modify the ```mimic_data_dir``` variable to the path of MIMIC-III folder in the ```./preprocess/preprocess_mimic_iii_large.py``` file, and run

      ```bash
      cd preprocess
      python preprocess_mimic_iii_large.py
      ```

   2. (opt.) If you want to build the **CIP** task, it is necessary to obtain a file all_hourly_data.h5 first. Otherwise, you can skip this step. 
 
      1) Start by referring to the [MIT-LCP](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres) repository to create a database where you can host the MIMIC-III data.
      2) Next, you will need to generate the file named ```all_hourly_data.h5``` using the [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract). Once generated, place the ```all_hourly_data.h5``` file in the ```save_data_folder/tmp/``` directory. You can locate the ```interv_outPath``` variable in the ```split_data_preprocessing_large.py``` file and ensure that it points to the correct destination, i.e., the ```save_data_folder/tmp/``` folder.

   3. Modify the ```data_root_folder``` and ```mimic_data_dir``` variables to the MIMIC-III folder path in the ```split_data_preprocessing_large.py```, and run the following command for downstream tasks generation:
      ```bash
      python split_data_preprocessing_large.py
      ```


## Run the Model

   Please refer to the script [run.sh](./run.sh) for specific configurations used to obtain the main experimental results in the paper. To reproduce the results, please uncomment the corresponding commands and run:
   ```bash
   bash run.sh
   ```


   Example:
   ```bash
   python Main_warp.py --task {task_name} \
                  --data_path {path_to_data_folder} \
                  --log {log_path} \
                  --save_path {save_path}
                  --epoch {epoch} \
                  --seed {seed} \
                  --lr 0.001 \
                  --batch {batch_size} \
                  --warp_num {warp_num} \
                  --dp_flag
   ```

   - ```task```: the downstram task name, select from ```[mor, decom, cip, wbm, los, active, physio]```.
   - ```seed```: the seed for parameter initialization.
   - ```warp_num```: customize $L^{(n)}$ for each warp layer. The first layer is always 0, and each layer is split by '_', e.g., ```'0_12_6'``` (unnormalized version) or ```'0_0.2_1.2'``` (normalized version).
   - ```dp_flag```: use DataParallel for training or not.


   For more details, please refer to [run.sh](./run.sh) and [Main_warp.py](./Main_warp.py).

## License

The original [MIMIC-III database](https://mimic.mit.edu/docs/iii/) is hosted and maintained on [PhysioNet](https://physionet.org/about/) under [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciii/view-license/1.4/), and is publicly accessible at [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/).

Our code in this repository is licensed under the [MIT license](./LICENSE).
