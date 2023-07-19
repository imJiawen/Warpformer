import sys
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import math
CHUNK_KEY = {'ONSET': 0, 'CONTROL': 1, 'ON_INTERVENTION': 2, 'WEAN': 3}

def create_mor_large(adm, events, adm2subj_dict, feature_map, filt_adm_ids=None):
    mor_adm_icu_id = []
    mor_data = []
    mor_label = []
    num_type = 122

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        adm_id = int(p_row.HADM_ID)
        if filt_adm_ids is not None and adm_id not in filt_adm_ids:
            continue

        
        p = events.loc[events.HADM_ID.isin([adm_id])]

        in_time = p.CHARTTIME.min()
        p = p.loc[(p.CHARTTIME-in_time)<=pd.Timedelta(48,'h')]
        
        if p.shape[0] < 1:
            continue
            
        patient =  [[] for _ in range(len(feature_map))]
        for _, row in p.iterrows():
            if row.NAME in feature_map:
                patient[feature_map[row.NAME]].append((row.CHARTTIME, row.VALUENUM))
        
        if adm_id in adm2subj_dict:
            mor_adm_icu_id.append((adm2subj_dict[adm_id], adm_id, None))
        else:
            mor_adm_icu_id.append((None, adm_id, None))
        mor_data.append(patient)
        mor_label.append(int(p_row.HOSPITAL_EXPIRE_FLAG))

    return mor_adm_icu_id, mor_data, mor_label

def create_decompensation_large(adm, events, feature_map, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, sample_rate=1.0, shortest_length=4.0, future_time_interval=24.0, filt_adm_ids=None):
    eps=1e-6
    adm_icu_id = []
    decom_data = []
    decom_label = []

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        adm_id = int(p_row.HADM_ID)
        if filt_adm_ids is not None and adm_id not in filt_adm_ids:
            continue

        # empty label
        if pd.isnull(p_row.HOSPITAL_EXPIRE_FLAG) or adm_id not in icu_dict:
            continue
        
        mortality = p_row.HOSPITAL_EXPIRE_FLAG
        
        p = events.loc[events.HADM_ID.isin([adm_id])]
        

        for icustay_id in icu_dict[adm_id]:
            icu_data = []

            if pd.isnull(los_dict[str(adm_id)+'_'+str(icustay_id)]):
                #print("(length of stay is missing)", adm_ids[i], icustay_id)
                continue

            los = 24.0 * los_dict[str(adm_id)+'_'+str(icustay_id)]  # in hours

            if adm_id in adm2deathtime_dict:
                deathtime = adm2deathtime_dict[adm_id]
            else:
                deathtime = None
                
            intime = pd.to_datetime(icu_dict[adm_id][icustay_id][0])
            outtime = pd.to_datetime(icu_dict[adm_id][icustay_id][1])

            # ############## for check ##################

            if deathtime is None:
                lived_time = 1e18
            else:
                lived_time = (pd.to_datetime(deathtime) - intime).total_seconds() / 3600.0

                
            p_icu_data = p.loc[((p.CHARTTIME>intime)&(p.CHARTTIME<outtime))]

            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate) 
            sample_times = list(filter(lambda x: x > shortest_length, sample_times))

            for t in sample_times:
                # get label
                if mortality == 0:
                    cur_mortality = 0
                else:
                    cur_mortality = int(lived_time - t < future_time_interval)
                
                # get data
                p_sample = p_icu_data.loc[(pd.Timedelta(t - shortest_length,'h')<=(p_icu_data.CHARTTIME-intime))&((p_icu_data.CHARTTIME-intime)<=pd.Timedelta(t,'h'))]
                
                if p_sample.shape[0] < 1:
                    continue
                
                count = 0
                patient_icu_sample =  [[] for _ in range(len(feature_map))]

                for _, row in p_sample.iterrows():
                    if row.NAME in feature_map:
                        patient_icu_sample[feature_map[row.NAME]].append((row.CHARTTIME, row.VALUENUM))
                        count += 1
                        
                if count < 1:
                    continue

                adm_icu_id.append((adm2subj_dict[adm_id], adm_id, icustay_id))
                decom_data.append(patient_icu_sample)
                decom_label.append(cur_mortality) 


    print("Number of created samples:", len(decom_data), len(adm_icu_id), len(decom_label))
    print("Number of features:", len(decom_data[0]))
    return adm_icu_id, decom_data, decom_label

def create_los_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, sample_rate=1.0, shortest_length=4.0, filt_adm_ids=None):
    eps=1e-6
    adm_icu_id = []
    decom_data = []
    decom_label = []

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        adm_id = int(p_row.HADM_ID)
        if filt_adm_ids is not None and adm_id not in filt_adm_ids:
            continue
        
        # empty label
        if pd.isnull(p_row.HOSPITAL_EXPIRE_FLAG) or adm_id not in icu_dict:
            continue
        
        p = events.loc[events.HADM_ID.isin([adm_id])]

        for icustay_id in icu_dict[adm_id]:
            icu_data = []

            if pd.isnull(los_dict[str(adm_id)+'_'+str(icustay_id)]):
                continue

            los = 24.0 * los_dict[str(adm_id)+'_'+str(icustay_id)]  # in hours


            intime = pd.to_datetime(icu_dict[adm_id][icustay_id][0])
            outtime = pd.to_datetime(icu_dict[adm_id][icustay_id][1])

            p_icu_data = p.loc[((p.CHARTTIME>intime)|(p.CHARTTIME<outtime))]

            sample_times = np.arange(0.0, los + eps, sample_rate) 
            sample_times = list(filter(lambda x: x > shortest_length, sample_times))

            for t in sample_times:

                # get data
                p_sample = p_icu_data.loc[(pd.Timedelta(t-shortest_length,'h')<=(p_icu_data.CHARTTIME-intime))&((p_icu_data.CHARTTIME-intime) < pd.Timedelta(t,'h'))]
                
                if p_sample.shape[0] < 1:
                    continue
                    
                patient_icu_sample =  [[] for _ in range(len(feature_map))]
                for _, row in p_sample.iterrows():
                    if row.NAME in feature_map:
                        patient_icu_sample[feature_map[row.NAME]].append((row.CHARTTIME, row.VALUENUM))
                    

                cur_label = number2cls((los - t)/24)
                adm_icu_id.append((adm2subj_dict[adm_id], adm_id, icustay_id))
                decom_data.append(patient_icu_sample)
                decom_label.append(cur_label) 
                
    print("Number of created samples:", len(decom_data), len(adm_icu_id), len(decom_label))
    print("Number of features:", len(decom_data[0]))
    return adm_icu_id, decom_data, decom_label

def create_wbm_large(adm, events, feature_map, chart_label_dict, icu_dict, los_dict, adm2deathtime_dict, adm2subj_dict, sample_rate=1.0, observ_win=4.0, future_time_interval=1.0, filt_adm_ids=None):
    eps=1e-6
    adm_icu_id = []
    wbm_data = []
    wbm_label = []

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        adm_id = int(p_row.HADM_ID)
        if filt_adm_ids is not None and adm_id not in filt_adm_ids:
            continue
        
        # empty label
        if adm_id not in icu_dict:
            continue
        
        p = events.loc[events.HADM_ID.isin([adm_id])]
        
        
        for icustay_id in icu_dict[adm_id]:

            if pd.isnull(los_dict[str(adm_id)+'_'+str(icustay_id)]):
                continue

            los = 24.0 * los_dict[str(adm_id)+'_'+str(icustay_id)]  # in hours

            if adm_id in adm2deathtime_dict:
                deathtime = adm2deathtime_dict[adm_id]
            else:
                deathtime = None
            
            intime = pd.to_datetime(icu_dict[adm_id][icustay_id][0])
            outtime = pd.to_datetime(icu_dict[adm_id][icustay_id][1])


            if deathtime is None:
                lived_time = (outtime - intime).total_seconds() / 3600.0
            else:
                lived_time = (pd.to_datetime(deathtime) - intime).total_seconds() / 3600.0

            # select data within icu stay
            p_icu_data = p.loc[((p.CHARTTIME>intime)|(p.CHARTTIME<outtime))]


            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate) 
            # start point of observation
            sample_times = list(filter(lambda x: x < (min(los, lived_time) - observ_win), sample_times))

            for t in sample_times:

                vital_label = [0]*len(chart_label_dict)
                
                # get data
                p_sample = p_icu_data.loc[(pd.Timedelta(t,'h')<=(p_icu_data.CHARTTIME-intime))&((p_icu_data.CHARTTIME-intime) < pd.Timedelta(t+observ_win,'h'))]
                if p_sample.shape[0] < 1:
                    continue
                patient_icu_sample =  [[] for _ in range(len(feature_map))]
                for _, row in p_sample.iterrows():
                    if row.NAME in feature_map:
                        patient_icu_sample[feature_map[row.NAME]].append((row.CHARTTIME, row.VALUENUM))
                    
                p_label = p_icu_data.loc[(pd.Timedelta(t + observ_win,'h')<=(p_icu_data.CHARTTIME-intime))&((p_icu_data.CHARTTIME-intime) < pd.Timedelta(t + observ_win + future_time_interval,'h'))]
                
                for _, row in p_label.iterrows():
                    if row.NAME in chart_label_dict:
                        vital_label[chart_label_dict[row.NAME]] = 1
                        
                adm_icu_id.append((adm2subj_dict[adm_id], adm_id, icustay_id))
                wbm_data.append(patient_icu_sample)
                wbm_label.append(vital_label) 
                
    print("Number of created samples:", len(wbm_data), len(adm_icu_id), len(wbm_label))
    print("Number of features:", len(wbm_data[0]))
    return adm_icu_id, wbm_data, wbm_label

def create_interv_pred_large(adm, events, feature_map, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict, interv, sample_rate=6, observ_win=6, future_time_interval=4, gap_win=6, filt_adm_ids=None):
    eps=1e-6
    adm_icu_id = []
    ip_data = []
    vent_labels = []
    vaso_labels = []

    # go through icu stays
    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        adm_id = int(p_row.HADM_ID)
        if filt_adm_ids is not None and adm_id not in filt_adm_ids:
            continue
        
        # empty label
        if adm_id not in icu_dict:
            continue
        
        p = events.loc[events.HADM_ID.isin([adm_id])]

        for icustay_id in icu_dict[adm_id]:

            if pd.isnull(los_dict[str(adm_id)+'_'+str(icustay_id)]):
                continue

            adm_index = (adm2subj_dict[adm_id], adm_id, icustay_id)

            los = 24.0 * los_dict[str(adm_id)+'_'+str(icustay_id)]  # in hours

            if adm_id in adm2deathtime_dict:
                deathtime = adm2deathtime_dict[adm_id]
            else:
                deathtime = None
                
            intime = pd.to_datetime(icu_dict[adm_id][icustay_id][0])
            outtime = pd.to_datetime(icu_dict[adm_id][icustay_id][1])

            if deathtime is None:
                lived_time = (outtime - intime).total_seconds() / 3600.0
            else:
                lived_time = (pd.to_datetime(deathtime) - intime).total_seconds() / 3600.0

            # select data within icu stay
            p_icu_data = p.loc[((p.CHARTTIME>intime)|(p.CHARTTIME<outtime))]

            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate) 
            sample_times = list(filter(lambda x: x < (min(los, lived_time) - observ_win), sample_times))

            for t in sample_times:
                # get data
                # get data
                p_sample = p_icu_data.loc[(pd.Timedelta(t,'h')<=(p_icu_data.CHARTTIME-intime))&((p_icu_data.CHARTTIME-intime) < pd.Timedelta(t+observ_win,'h'))]
                if p_sample.shape[0] < 1:
                    continue
                patient_icu_sample =  [[] for _ in range(len(feature_map))]
                for _, row in p_sample.iterrows():
                    if row.NAME in feature_map:
                        patient_icu_sample[feature_map[row.NAME]].append((row.CHARTTIME, row.VALUENUM))
                
                vent_label = cal_label(interv['vent'], adm_index, int(t), observ_win, gap_win, future_time_interval)
                vaso_label = cal_label(interv['vaso'], adm_index, int(t), observ_win, gap_win, future_time_interval)

                if vent_label is not None and vaso_label is not None:
                    adm_icu_id.append(adm_index)
                    ip_data.append(patient_icu_sample)
                    vent_labels.append(vent_label) 
                    vaso_labels.append(vaso_label) 

    print("Number of created samples:", len(ip_data), len(adm_icu_id), len(vent_labels))
    print("Number of features:", len(ip_data[0]))
    return adm_icu_id, ip_data, vent_labels, vaso_labels


def number2cls(number):
        
    if number <= 7.0:
        if number <= 0:
            number = 1
        return math.ceil(number)
    elif 7 < number < 14:
        return 8
    else:
        return 9

def cal_label(interv, adm_index, t, observ_win, gap_win, future_time_interval):
    if adm_index not in interv:
        return None
        
    y_patient = interv[adm_index].values

    result_window = y_patient[t+observ_win+gap_win : t+observ_win+gap_win+future_time_interval]

    result_window_diff = set(np.diff(result_window))
    gap_window = y_patient[t+observ_win:t+observ_win+gap_win]
    gap_window_diff = set(np.diff(gap_window))


    if 1 in gap_window_diff or -1 in gap_window_diff:
        result = None
    elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 0):
        result = CHUNK_KEY['CONTROL']
    elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 1):
        result = CHUNK_KEY['ON_INTERVENTION']
    elif 1 in result_window_diff: 
        result = CHUNK_KEY['ONSET']
    elif -1 in result_window_diff:
        result = CHUNK_KEY['WEAN']
    else:
        result = None

    return result