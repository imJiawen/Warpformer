import pandas as pd
from tqdm import tqdm, trange
import pickle
import numpy as np
import copy
import os
import random
random.seed(49297)

import proc_util
from proc_util.task_build import *
from proc_util.extract_cip_label import *

mimic_data_dir = 'path/to/mimic-iii-clinical-database-1.4/'

min_time_period = 48

def trim_los(data):
    """Used to build time set
    """
    num_features = len(data[0]) # final features (excluding EtCO2)
    max_length = 300  # maximum length of time stamp(48 * 60)
    a = np.zeros((len(data), num_features, max_length))
    timestamps = []

    for i in range(len(data)):
        
        TS = set()
        for j in range(num_features):
            for k in range(len(data[i][j])):
                TS.add(data[i][j][k][0].to_pydatetime())
                
        TS = list(TS)
        TS.sort()
        timestamps.append(TS)

        for j in range(len(data[i])):
            for t,v in data[i][j]:
                idx = TS.index(t.to_pydatetime())
                if idx < max_length:
                    a[i, j, idx] = v

    print("feature extraction success")
    print("value processing success ")
    return a, timestamps

def remove_missing_dim(x, M, T):
    new_x = np.zeros((len(x), len(x[0]), len(x[0][0])))
    new_M = np.zeros((len(M), len(M[0]), len(M[0][0])))
    new_T = [[] for _ in range(len(x))]
    
    tmp_x = x.sum(1).squeeze() # [B 1 L]
    for b in range(len(tmp_x)):
        new_l = 0
        for l in range(len(tmp_x[b])):
            if tmp_x[b][l] > 0:
                new_x[b,:,new_l] = x[b,:,l]
                new_M[b,:,new_l] = M[b,:,l]
                # new_T[b,:,new_l] = T[b,:,l]
                new_T[b].append(T[b][l])
                new_l += 1
                
    return new_x, new_M, new_T

def fix_input_format(x, T):
    """Return the input in the proper format
    x: observed values
    M: masking, 0 indicates missing values
    delta: time points of observation
    """
    timestamp = 200
    num_features = 122

    M = np.zeros_like(x)
    # x[x > 500] = 0.0
    x[x < 0] = 0.0
    M[x > 0] = 1
    
    x, M, T = remove_missing_dim(x, M, T)
    
    x = x[:, :, :timestamp]
    M = M[:, :, :timestamp]
    
    delta = np.zeros((x.shape[0], 1, x.shape[-1]))
    
    ts_len = []
    for i in range(len(T)):
        for j in range(1, len(T[i])):
            if j >= timestamp:
                break
            delta[i,0,j] = (T[i][j] - T[i][0]).total_seconds() / 3600.0
        ts_len.append(len(T[i]))

    return x, M, delta, ts_len

def preproc_xy(adm_icu_id, data_x, data_y, dataset_name, split_name):

    out_value, out_timestamps = trim_los(data_x)

    x, m, T, ts_len = fix_input_format(out_value, out_timestamps) 
    print("timestamps format processing success")
    
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)
    
    pickle.dump(adm_icu_id, open(dataset_name + split_name + '_sub_adm_icu_idx.p', 'wb'))
    # pickle.dump(ts_len, open(dataset_name + split_name + '_ts_len.p', 'wb'))
    save_xy(x, m, T, data_y, dataset_name + split_name)
    
def save_xy(in_x,in_m,in_T, label, save_path):
    in_T = np.expand_dims(in_T[:,0,:], axis=1)
    x = np.concatenate((in_x,in_m,in_T) , axis=1)  # input format
    y = np.array(label)
    np.save(save_path + '_input.npy', x)
    np.save(save_path + '_output.npy', y)
    print(x.shape)
    print(y.shape)

    print(save_path, " saved success")
    
def preproc_interv_xy(adm_icu_id, data_x, vent_label, vaso_label, dataset_name, split_name):

    out_value, out_timestamps = trim_los(data_x)

    x, m, T, ts_len = fix_input_format(out_value, out_timestamps)
    print("timestamps format processing success")
    
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    # pickle.dump(ts_len, open(dataset_name + split_name + '_ts_len.p', 'wb'))
    pickle.dump(adm_icu_id, open(dataset_name + split_name + '_sub_adm_icu_idx.p', 'wb'))
    save_interv_xy(x, m, T, vent_label, vaso_label, dataset_name + split_name)
    
def save_interv_xy(in_x,in_m,in_T, vent_label, vaso_label, save_path):
    in_T = np.expand_dims(in_T[:,0,:], axis=1)
    x = np.concatenate((in_x,in_m,in_T) , axis=1)  # input format
    vent_label = np.array(vent_label)
    vaso_label = np.array(vaso_label)
    
    np.save(save_path + '_input.npy', x)
    np.save(save_path + '_vent_output.npy', vent_label)
    np.save(save_path + '_vaso_output.npy', vaso_label)
    print(x.shape)
    print(vent_label.shape)
    print(vaso_label.shape)
    print(save_path, " saved success")


def create_map(icu,events):
    chart_label_dict = {}
    icu_dict = {}
    los_dict = {}
    adm2subj_dict = {}
    adm2deathtime_dict = {}

    for _,p_row in tqdm(icu.iterrows(), total=icu.shape[0]):
        if p_row.HADM_ID not in icu_dict:
            icu_dict.update({p_row.HADM_ID:{p_row.ICUSTAY_ID:[p_row.INTIME, p_row.OUTTIME]}})
            los_dict.update({str(p_row.HADM_ID)+'_'+str(p_row.ICUSTAY_ID) : p_row.LOS})
            
        elif p_row.ICUSTAY_ID not in icu_dict[p_row.HADM_ID]:
            icu_dict[p_row.HADM_ID].update({p_row.ICUSTAY_ID:[p_row.INTIME, p_row.OUTTIME]})
            los_dict.update({str(p_row.HADM_ID)+'_'+str(p_row.ICUSTAY_ID): p_row.LOS})
            
        if p_row.HADM_ID not in adm2subj_dict:
            adm2subj_dict.update({p_row.HADM_ID:p_row.SUBJECT_ID})

    for _,p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        if p_row.HADM_ID not in adm2deathtime_dict:
                adm2deathtime_dict.update({p_row.HADM_ID:p_row.DEATHTIME})
                
    # get feature set
    feature_set = []
    feature_map = {}
    events = events.loc[~(events.CHARTTIME.isna() & events.VALUENUM.isna())]

    idx = 0
    for i in events.NAME:
        if i not in feature_set:
            feature_map[i] = idx
            idx += 1
            feature_set.append(i)
    
    type_dict = {}
    for i in feature_set:
        tmp_p = events.loc[events.NAME.isin([i])]
        tmp_set = set(tmp_p.TABLE)
        type_dict.update({i: tmp_set})

    idx = 0
    for k in type_dict:
        if 'chart' in type_dict[k] or 'lab' in type_dict[k]:
            if k not in chart_label_dict and k != "Mechanical Ventilation":
                chart_label_dict[k] = idx
                idx += 1
            
    print("got ", str(len(feature_set)), " features")
    return feature_map, chart_label_dict, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict
    

if __name__ == '__main__':
    data_root_folder = "path/to/save_data_folder/"

    if not os.path.isdir(data_root_folder):
        os.mkdir(data_root_folder)

    data_tmp_folder = data_root_folder + "tmp/"

    if not os.path.isdir(data_tmp_folder):
        os.mkdir(data_tmp_folder)

    adm_id_folder =  "./adm_id/"

    bio_path = data_tmp_folder + "patient_records_large.p"
    interv_outPath = data_tmp_folder + "all_hourly_data.h5"
    resource_path = "./proc_util/resource/"
    
    print("Loading data...")
    icu = pd.read_csv(mimic_data_dir+'ICUSTAYS.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS'])
    icu.drop_duplicates(inplace=True)
    
    adm = pd.read_csv(mimic_data_dir+'ADMISSIONS.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'])
    adm.drop_duplicates(inplace=True)

    mimi_iii_event = './mimic_iii_events.csv'

    events = pd.read_csv(mimi_iii_event, usecols=['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VALUENUM', 'TABLE', 'NAME'])
    events.drop_duplicates(inplace=True)

    events = events.loc[~(events.CHARTTIME.isna() & events.VALUENUM.isna())]

    feature_map, chart_label_dict, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict = create_map(icu,events)
    
    # This is optional
    # remove_mod_idx = [] # keep 122 features
    remove_mod_idx = [2, 65, 91, 119, 42, 97, 120, 115, 94, 62, 105, 63, 73, 81, 87, 98, 110, 67, 93] # keep 103 features 
    # remove_mod_idx = [2, 66, 92, 120, 42, 98, 121, 116, 95, 63, 106, 64, 74, 82, 88, 99, 111, 68, 94]


    tmp_feature_name = sorted(feature_map.items(),  key=lambda d: d[1], reverse=False)

    feature_map = {}
    feature_name = []
    new_idx = 0
    for i,j in tmp_feature_name:
        if j not in remove_mod_idx:
            feature_map[i] = new_idx
            new_idx += 1
            feature_name.append(i)
    print("got ", str(len(feature_map)), " features")
    
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    adm.ADMITTIME = pd.to_datetime(adm.ADMITTIME)
    adm.DISCHTIME = pd.to_datetime(adm.DISCHTIME)
    adm.DEATHTIME = pd.to_datetime(adm.DEATHTIME)


    # data split
    train_adm_id = pickle.load(open(adm_id_folder + 'train_adm_idx.p', 'rb'))
    test_adm_id = pickle.load(open(adm_id_folder + 'test_adm_idx.p', 'rb'))
    val_adm_id = pickle.load(open(adm_id_folder + 'val_adm_idx.p', 'rb'))

    # # #==== Mortality ====
    print("Building Mortality task...")

    mor_adm_icu_id, mor_data, mor_label = create_mor_large(adm, events, adm2subj_dict, feature_map, filt_adm_ids=train_adm_id)
    preproc_xy(mor_adm_icu_id, mor_data, mor_label, data_root_folder+'mor/', 'train')

    mor_adm_icu_id, mor_data, mor_label = create_mor_large(adm, events, adm2subj_dict, feature_map, filt_adm_ids=test_adm_id)
    preproc_xy(mor_adm_icu_id, mor_data, mor_label, data_root_folder+'mor/', 'test')

    mor_adm_icu_id, mor_data, mor_label = create_mor_large(adm, events, adm2subj_dict, feature_map, filt_adm_ids=val_adm_id)
    preproc_xy(mor_adm_icu_id, mor_data, mor_label, data_root_folder+'mor/', 'val')
    
    print("Build Mortality task done")

    # #==== Decompensation ====
    print("Building Decompensation task...")
    
    adm_icu_id, decom_data, decom_label = create_decompensation_large(adm, events, feature_map, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, \
    sample_rate=12, shortest_length=24, future_time_interval=24.0, filt_adm_ids=train_adm_id)
    
    preproc_xy(adm_icu_id, decom_data, decom_label, data_root_folder+'decom/', 'train')
    
    adm_icu_id, decom_data, decom_label = create_decompensation_large(adm, events, feature_map, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, \
    sample_rate=12, shortest_length=24, future_time_interval=24.0, filt_adm_ids=test_adm_id)
    
    preproc_xy(adm_icu_id, decom_data, decom_label, data_root_folder+'decom/', 'test')
    
    adm_icu_id, decom_data, decom_label = create_decompensation_large(adm, events, feature_map, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, \
    sample_rate=12, shortest_length=24, future_time_interval=24.0, filt_adm_ids=val_adm_id)
    
    preproc_xy(adm_icu_id, decom_data, decom_label, data_root_folder+'decom/', 'val')
    
    print("Build Decompensation task done")
    
    # # #==== Length of Stay ====
    print("Building Length of Stay task...")
    
    adm_icu_id, los_data, los_label = create_los_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, \
    sample_rate=12, shortest_length=24, filt_adm_ids=train_adm_id)

    preproc_xy(adm_icu_id, los_data, los_label, data_root_folder+'los/', 'train')
    
    adm_icu_id, los_data, los_label = create_los_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, \
    sample_rate=12, shortest_length=24, filt_adm_ids=test_adm_id)

    preproc_xy(adm_icu_id, los_data, los_label, data_root_folder+'los/', 'test')
    
    adm_icu_id, los_data, los_label = create_los_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, \
    sample_rate=12, shortest_length=24, filt_adm_ids=val_adm_id)

    preproc_xy(adm_icu_id, los_data, los_label, data_root_folder+'los/', 'val')

    print("Build Length of Stay task done")
    
    # #==== Next Timepoint Will be Measured ====
    print("Building Next Timepoint Will be Measured task...")
    
    wbm_adm_icu_id, wbm_data, wbm_label = create_wbm_large(adm, events, feature_map, chart_label_dict, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, \
     sample_rate=12.0, observ_win=48.0, future_time_interval=1.0, filt_adm_ids=train_adm_id)

    preproc_xy(wbm_adm_icu_id, wbm_data, wbm_label, data_root_folder+'wbm/', 'train')
    
    wbm_adm_icu_id, wbm_data, wbm_label = create_wbm_large(adm, events, feature_map, chart_label_dict, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, \
     sample_rate=12.0, observ_win=48.0, future_time_interval=1.0, filt_adm_ids=test_adm_id)

    preproc_xy(wbm_adm_icu_id, wbm_data, wbm_label, data_root_folder+'wbm/', 'test')
    
    wbm_adm_icu_id, wbm_data, wbm_label = create_wbm_large(adm, events, feature_map, chart_label_dict, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, \
     sample_rate=12.0, observ_win=48.0, future_time_interval=1.0, filt_adm_ids=val_adm_id)

    preproc_xy(wbm_adm_icu_id, wbm_data, wbm_label, data_root_folder+'wbm/', 'val')

    print("Build Next Timepoint Will be Measured task done")
    
    # #==== Clinical Intervention Prediction ====
    print("Building Clinical Intervention Prediction task...")
    
    if not os.path.isfile(interv_outPath):
        # extract_cip_data(resource_path, interv_outPath, dbname, host, user, password)
        print("Cannot find file ", interv_outPath)
        print("Please obtain all_hourly_data.h5 from https://github.com/MLforHealth/MIMIC_Extract first.")
        sys.exit(0)
        
    
    print("Loading files from ", interv_outPath)
    Y = pd.read_hdf(interv_outPath,'interventions')
    Y = Y[['vent', 'vaso']]
    
    cip_adm_icu_id, cip_data, vent_labels, vaso_labels = create_interv_pred_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict, Y, \
    sample_rate=6, observ_win=6, future_time_interval=4, gap_win=6, filt_adm_ids=train_adm_id)
    
    preproc_interv_xy(cip_adm_icu_id, cip_data, vent_labels, vaso_labels, data_root_folder+'cip/', "train")
    
    cip_adm_icu_id, cip_data, vent_labels, vaso_labels = create_interv_pred_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict, Y, \
    sample_rate=6, observ_win=6, future_time_interval=4, gap_win=6, filt_adm_ids=test_adm_id)
    
    preproc_interv_xy(cip_adm_icu_id, cip_data, vent_labels, vaso_labels, data_root_folder+'cip/', "test")
    
    cip_adm_icu_id, cip_data, vent_labels, vaso_labels = create_interv_pred_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict, Y, \
    sample_rate=6, observ_win=6, future_time_interval=4, gap_win=6, filt_adm_ids=val_adm_id)
    
    preproc_interv_xy(cip_adm_icu_id, cip_data, vent_labels, vaso_labels, data_root_folder+'cip/', "val")
    
    
    print("Build all the task done.")