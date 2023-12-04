# %%
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATH_TO_RESPONSES = 'responses'
PATH_TO_MASKING = r'C:\Users\matt\AppData\LocalLow\DefaultCompany\HololensAcoustics'

# %%

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def append_row(df, row):
    return pd.concat([
                df, 
                pd.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)

    
        

def do_localisation(localisation_dict, df):
    print(localisation_dict.keys())
    print(localisation_dict['tasks'][0])

    actual_source = localisation_dict['tasks'][0]['sourcePosition']
    
    selected_source = localisation_dict['tasks'][0]['selectedSource']

    actual_source = actual_source[0:2]
    selected_source = selected_source[0:2]

    fig, ax = plt.subplots()

    ax.plot([0, actual_source[0]], [0, actual_source[1]])
    ax.plot([0, selected_source[0]], [0, selected_source[1]])
    angle = np.degrees(angle_between(actual_source, selected_source))
    plt.show()

    print(angle)
    row = pd.Series({'ParticipantID': localisation_dict['ParticipantID'],
                     'ActualSource': actual_source,
                     'SelectedSource': selected_source,
                     'AngularDistance': angle})
    df = append_row(df, row)
# %%

localisation_df = pd.DataFrame(columns=['ParticipantID', 'ArrayDistance',
                                         'ActualSource', 'SelectedSource',
                                         'AngularDistance', 'Timestamp',
                                         'ClipName', 'RT'])


for json_file in Path(PATH_TO_RESPONSES).glob('*.json'):
    file_id = json_file.stem.split('_')
    task_dict = json.load(open(json_file))
    participant_id = file_id.pop(0)
    tasks = task_dict['tasks']

    if 'localisation' in file_id[0]:
        # do_localisation(task_dict, df)
        if len(tasks) != 27:
            continue

        array_distance = file_id[2][0:-5]
        for task in tasks:
            
            source_position = task['sourcePosition']
            selected_source = task['selectedSource']
            angle = np.degrees(angle_between(source_position, selected_source))
            timestamp = task['timeStamp']
            clip_name = task['clip_name']
            rt = task['rt']
            row = pd.Series(
                {'ParticipantID': task_dict['ParticipantID'],
                 'ArrayDistance': array_distance,
                 'ActualSource': source_position,
                 'SelectedSource': selected_source,
                 'AngularDistance': angle,
                 'Timestamp': timestamp,
                 'ClipName': clip_name,
                 'RT': rt      
                 })
            localisation_df = append_row(localisation_df, row)

    


# df.to_excel('test.xlsx')

# %%


masking_df = pd.DataFrame(columns=['ParticipantID', 'ListenerPosition',
                                         'AngularDistance', 'SoundSourcePositions',
                                         'Timestamp', 'Response', 'RT'])
for json_file in Path(PATH_TO_MASKING).glob('*.json'):
    file_id = json_file.stem.split('_')
    task_dict = json.load(open(json_file))
    participant_id = file_id.pop(0)
    tasks = task_dict['tasks']

    if 'masking' in file_id[0]:
        # do_localisation(task_dict, df)
        # if len(tasks) != 27:
        #     continue

        for task in tasks:
            if 'rt' not in task.keys():
                continue
            listener_position = task['listenerPosition']
            angle = task['angularDistance']
            source_positions = task['soundSourcePositions']
            timestamp = task['timeStamp']
            response = task['response']
            rt = task['rt']
            row = pd.Series(
                {'ParticipantID': task_dict['ParticipantID'],
                 'ListenerPosition': listener_position,
                 'AngularDistance': angle,
                 'SoundSourcePositions': source_positions,
                 'Timestamp': timestamp,
                 'Response': response,
                 'RT': rt      
                 })
            masking_df = append_row(masking_df, row)


    

# %%
