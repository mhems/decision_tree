#!/usr/bin/python3

import pandas as pd
import numpy as np
import tables
import glob
from StringIO import StringIO

# Aggregates data from all json files into one csv summary

def gen_dataframe(path):
    """Given path to data, prints song info summary"""
    print("ID, num_bars,avg_bar_len,std_bar_len,avg_bar_conf,std_bar_conf," \
              "num_beats,avg_beat_len,std_beat_len,avg_beat_conf,std_beat_conf," \
              "num_tatums,avg_tatum_len,std_tatum_len,avg_tatum_conf,std_tatum_conf," \
              "num_sections,avg_section_len,std_section_len,avg_section_conf,std_section_conf,"\
              "duration,key_val,key_conf,tempo_val,tempo_conf,genre")
    for songpath in glob.glob(path + 'echonest_features/*/*.json'):
        printRow(get_SALAMI_song_info(songpath))


def get_SALAMI_song_info(songpath):
    """Returns tuple of song info at songpath"""
    df = pd.read_json(songpath, typ='series')
    metadata = df['metadata']
    id = metadata['identifier']
    bars_tuple     = collect_stats(df['bars'])
    beats_tuple    = collect_stats(df['beats'])
    tatums_tuple   = collect_stats(df['tatums'])
    sections_tuple = collect_stats(df['sections'])
    duration = metadata['duration']
    key_val = metadata['key']['value']
    key_conf = metadata['key']['confidence']
    tempo_val = metadata['tempo']['value']
    tempo_conf = metadata['tempo']['confidence']
    genre = "none"
    if 'genre' in metadata.keys() and metadata['genre'] != "":
        genre = metadata['genre']
    ret = [id]
    ret.extend(bars_tuple)
    ret.extend(beats_tuple)
    ret.extend(tatums_tuple)
    ret.extend(sections_tuple)
    ret.append(duration)
    ret.append(key_val)
    ret.append(key_conf)
    ret.append(tempo_val)
    ret.append(tempo_conf)
    ret.append(genre)
    return tuple(ret)

def collect_stats(dictionary):
    """Returns tuple of basic statistics over dictionary"""
    num = len(dictionary['start'])
    if num == 0:
        return None
    avg_len  = np.mean(dictionary['durations'])
    len_std  = np.std(dictionary['durations'])
    if None not in dictionary['confidence']:
        avg_conf = np.mean(dictionary['confidence'])
        conf_std = np.std(dictionary['confidence'])
    else:
        avg_conf = -1
        conf_std = -1
    return (num, avg_len, len_std, avg_conf, conf_std)

def printRow(tup):
    if "nan" not in tup:
        print(','.join([repr(e) for e in tup]))
    
if __name__ == '__main__':
    # send in path to SALAMI data
    gen_dataframe()
