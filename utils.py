import numpy as np
import pandas as pd


def process_raster(times, neurons):

    raster = pd.DataFrame()
    raster['neuron'] = neurons
    raster['times'] = times

    trains = []
    isi_series = []
    isis = []

    for neuron_id in raster['neuron'].value_counts().index.values:
        train = raster[raster['neuron'] == neuron_id]['times'].values
        isi_series.append(np.diff(train))
        isis += list(np.diff(train))
        trains.append(train)

    return trains, isi_series, isis


def save_isis_to_file(isi_series_list, file):

    with open(file, 'w') as outfile:
        for isi_series in isi_series_list:
            outfile.write(', '.join(list(map(str, isi_series))))
            outfile.write('\n')
