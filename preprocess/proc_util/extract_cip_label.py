from __future__ import print_function, division

# MIMIC IIIv14 on postgres 9.4
import os
import pandas as pd

from os.path import isfile, isdir, splitext
import datapackage
from proc_util.mimic_querier import *

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# SQL_DIR = os.path.join(CURRENT_DIR, 'SQL_Queries')
# STATICS_QUERY_PATH = os.path.join(SQL_DIR, 'statics.sql')


# SQL command params

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']


def load_datapackage_schema(json_fpath, resource_id=0):
    """ Load schema object

    Returns
    -------
    schema : schema object, with attributes
        field_names
        fields : list of dict
            Each dict provides info about the field (data type, etc)
    """
    spec = datapackage.DataPackage(json_fpath)
    schema = spec.resources[resource_id].schema
    return schema


def sanitize_df(data_df, schema, setup_index=True, missing_column_procedure='fill_zero'):
    """ Sanitize dataframe according to provided schema

    Returns
    -------
    data_df : pandas DataFrame
        Will have fields provided by schema
        Will have field types (categorical, datetime, etc) provided by schema.
    """
    data_df = data_df.reset_index()
    for ff, field_name in enumerate(schema.field_names):
        type_ff = schema.fields[ff].descriptor['type']
        if field_name not in data_df.columns:
            if missing_column_procedure == 'fill_zero':
                if type_ff == 'integer':
                    data_df[field_name] = 0
                elif type_ff == 'number':
                    data_df[field_name] = 0.0

    # Reorder columns to match schema
    data_df = data_df[schema.field_names]
    # Cast fields to required type (categorical / datetime)
    for ff, name in enumerate(schema.field_names):
        ff_spec = schema.descriptor['fields'][ff]
        if 'pandas_dtype' in ff_spec and ff_spec['pandas_dtype'] == 'category':
            data_df[name] = data_df[name].astype('category')
        elif 'type' in ff_spec and ff_spec['type'] == 'datetime':
            data_df[name] = pd.to_datetime(data_df[name])
    if hasattr(schema, 'primary_key'):
        data_df = data_df.sort_values(schema.primary_key)
        if setup_index:
            data_df = data_df.set_index(schema.primary_key)
    return data_df


def add_outcome_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()

    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hours, 'on':on_vals}) #icustay_id': icustay_id})


def add_blank_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    #icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0]*len(hrs))
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hrs, 'on':vals})#'icustay_id': icustay_id,

def continuous_outcome_processing(out_data, data, icustay_timediff):
    """

    Args
    ----
    out_data : pd.DataFrame
        index=None
        Contains subset of icustay_id corresp to specific sessions where outcome observed.
    data : pd.DataFrame
        index=icustay_id
        Contains full population of static demographic data

    Returns
    -------
    out_data : pd.DataFrame
    """
    out_data['intime'] = out_data['icustay_id'].map(data['intime'].to_dict())
    out_data['outtime'] = out_data['icustay_id'].map(data['outtime'].to_dict())
    out_data['max_hours'] = out_data['icustay_id'].map(icustay_timediff)
    out_data['starttime'] = out_data['starttime'] - out_data['intime']
    out_data['starttime'] = out_data.starttime.apply(lambda x: x.days*24 + x.seconds//3600)
    out_data['endtime'] = out_data['endtime'] - out_data['intime']
    out_data['endtime'] = out_data.endtime.apply(lambda x: x.days*24 + x.seconds//3600)
    out_data = out_data.groupby(['icustay_id'])

    return out_data


def save_outcome(data, querier, outcome_schema, host=None):
    """ Retrieve outcomes from DB and save to disk

    Vent and vaso are both there already - so pull the start and stop times from there! :)

    Returns
    -------
    Y : Pandas dataframe
        Obeys the outcomes data spec
    """
    icuids_to_keep = get_values_by_name_from_df_column_or_index(data, 'icustay_id')
    icuids_to_keep = set([str(s) for s in icuids_to_keep])

    # Add a new column called intime so that we can easily subtract it off
    data = data.reset_index()
    data = data.set_index('icustay_id')
    data['intime'] = pd.to_datetime(data['intime']) #, format="%m/%d/%Y"))
    data['outtime'] = pd.to_datetime(data['outtime'])
    icustay_timediff_tmp = data['outtime'] - data['intime']
    icustay_timediff = pd.Series([timediff.days*24 + timediff.seconds//3600
                                  for timediff in icustay_timediff_tmp], index=data.index.values)
    query = """
    select i.subject_id, i.hadm_id, v.icustay_id, v.ventnum, v.starttime, v.endtime
    FROM icustay_detail i
    INNER JOIN ventilation_durations v ON i.icustay_id = v.icustay_id
    where v.icustay_id in ({icuids})
    and v.starttime between intime and outtime
    and v.endtime between intime and outtime;
    """

    old_template_vars = querier.exclusion_criteria_template_vars
    querier.exclusion_criteria_template_vars = dict(icuids=','.join(icuids_to_keep))

    vent_data = querier.query(query_string=query)
    vent_data = continuous_outcome_processing(vent_data, data, icustay_timediff)
    vent_data = vent_data.apply(add_outcome_indicators)
    vent_data.rename(columns = {'on':'vent'}, inplace=True)
    vent_data = vent_data.reset_index()

    # Get the patients without the intervention in there too so that we
    ids_with = vent_data['icustay_id']
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = (ids_all - ids_with)
    #ids_without = map(int, ids_without)

    # Create a new fake dataframe with blanks on all vent entries
    out_data = data.copy(deep=True)
    out_data = out_data.reset_index()
    out_data = out_data.set_index('icustay_id')
    out_data = out_data.iloc[out_data.index.isin(ids_without)]
    out_data = out_data.reset_index()
    out_data = out_data[['subject_id', 'hadm_id', 'icustay_id']]
    out_data['max_hours'] = out_data['icustay_id'].map(icustay_timediff)

    # Create all 0 column for vent
    out_data = out_data.groupby('icustay_id')
    out_data = out_data.apply(add_blank_indicators)
    out_data.rename(columns = {'on':'vent'}, inplace=True)
    out_data = out_data.reset_index()

    # Concatenate all the data vertically
    Y = pd.concat([vent_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent']],
                   out_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent']]],
                  axis=0)

    # Start merging all other interventions
    table_names = [
        'vasopressor_durations',
        'adenosine_durations',
        'dobutamine_durations',
        'dopamine_durations',
        'epinephrine_durations',
        'isuprel_durations',
        'milrinone_durations',
        'norepinephrine_durations',
        'phenylephrine_durations',
        'vasopressin_durations'
    ]
    column_names = ['vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 
                    'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']

    # TODO(mmd): This section doesn't work. What is its purpose?
    for t, c in zip(table_names, column_names):
        # TOTAL VASOPRESSOR DATA
        query = """
        select i.subject_id, i.hadm_id, v.icustay_id, v.vasonum, v.starttime, v.endtime
        FROM icustay_detail i
        INNER JOIN {table} v ON i.icustay_id = v.icustay_id
        where v.icustay_id in ({icuids})
        and v.starttime between intime and outtime
        and v.endtime between intime and outtime;
        """
        new_data = querier.query(query_string=query, extra_template_vars=dict(table=t))
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns={'on': c}, inplace=True)
        new_data = new_data.reset_index()
        # c may not be in Y if we are only extracting a subset of the population, in which c was never
        # performed.
        if not c in new_data:
            print("Column ", c, " not in data.")
            continue

        Y = Y.merge(
            new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', c]],
            on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
            how='left'
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[c] = Y[c].astype(int)
        #Y = Y.sort_values(['subject_id', 'icustay_id', 'hours_in']) #.merge(df3,on='name')
        Y = Y.reset_index(drop=True)
        print('Extracted ' + c + ' from ' + t)


    tasks=["colloid_bolus", "crystalloid_bolus", "nivdurations"]

    for task in tasks:
        if task=='nivdurations':
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.starttime, v.endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.starttime between intime and outtime
            and v.endtime between intime and outtime;
            """
        else:
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.charttime AS starttime, 
                   v.charttime AS endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.charttime between intime and outtime
            """

        new_data = querier.query(query_string=query, extra_template_vars=dict(table=task))
        if new_data.shape[0] == 0: continue
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns = {'on':task}, inplace=True)
        new_data = new_data.reset_index()
        Y = Y.merge(
            new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', task]],
            on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
            how='left'
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[task] = Y[task].astype(int)
        Y = Y.reset_index(drop=True)
        print('Extracted ' + task)

    querier.exclusion_criteria_template_vars = old_template_vars

    Y = Y.filter(items=['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent'] + column_names + tasks)
    Y.subject_id = Y.subject_id.astype(int)
    Y.icustay_id = Y.icustay_id.astype(int)
    Y.hours_in = Y.hours_in.astype(int)
    Y.vent = Y.vent.astype(int)
    Y.vaso = Y.vaso.astype(int)
    y_id_cols = ID_COLS + ['hours_in']
    Y = Y.sort_values(y_id_cols)
    Y.set_index(y_id_cols, inplace=True)

    print('Shape of Y : ', Y.shape)

    # Turn back into columns
    df = Y.reset_index()
    df = sanitize_df(df, outcome_schema) 

    return df


def extract_cip_data(resource_path, outPath, dbname, schema_name, host, user, password):
    mimic_mapping_filename = os.path.join(resource_path, 'itemid_to_variable_map.csv')
    range_filename = os.path.join(resource_path, 'variable_ranges.csv')

    # Load specs for output tables
    static_data_schema = load_datapackage_schema(
        os.path.join(resource_path, 'static_data_spec.json'))
    outcome_data_schema = load_datapackage_schema(
        os.path.join(resource_path, 'outcome_data_spec.json'))

    query_args = {'dbname': dbname}
    query_args['host'] = host
    query_args['user'] = user
    query_args['password'] = password

    querier = MIMIC_Querier(query_args=query_args, schema_name="public,mimiciii")

    #############
    # Population extraction
    data = None
    
    print("Building data from scratch.")
    pop_size_string = ''

    min_age_string = 0
    min_dur_string = 12
    max_dur_string = 240
    min_day_string = str(float(12)/24)

    template_vars = dict(
        limit=pop_size_string, min_age=min_age_string, min_dur=min_dur_string, max_dur=max_dur_string,
        min_day=min_day_string
    )

    data_df = querier.query(query_file="./preprocess/proc_util/resource/statics.sql", extra_template_vars=template_vars)
    data = sanitize_df(data_df, static_data_schema, setup_index=False)

    if data is None: print('SKIPPED static_data')
    else:
        # So all subsequent queries will limit to just that already extracted in data_df.
        querier.add_exclusion_criteria_from_df(data, columns=['hadm_id', 'subject_id'])
        print("loaded static_data")

    #############
    # If there is outcome extraction
    print("Saving Outcomes...")
    Y = save_outcome(data, querier, outcome_data_schema, host=host)

    if Y is not None: print("Outcomes", Y.shape, Y.index.names, Y.columns.names, Y.columns)

    print(data.shape, data.index.names, data.columns.names)


    # data = data[data.index.get_level_values('icustay_id')]
    # data = data.reset_index().set_index(ID_COLS)


    print('Shape of Y : ', Y.shape)

    Y.to_hdf(outPath, 'interventions')

    print('extarct intervention label done!')
    print('data is save to ', outPath)
