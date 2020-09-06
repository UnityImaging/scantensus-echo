import os
import csv

import numpy as np
import pandas as pd

import scipy.interpolate
import scipy.signal

import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt


PLOT = True
INPUT_DIR = "/Users/matthew/Downloads/csv3"
CSV_FILE = "/Volumes/Matt-Data/strain3.csv"


def filter_lens(d):
    len_all = len(d)
    x_all = np.arange(len_all)
    mask = np.bitwise_not(np.isnan(d))

    x = x_all[mask]
    y = d[mask]
    w = np.ones_like(x)

    tck = scipy.interpolate.splrep(x=x, y=y, w=w, k=3, s=1)
    y_all = scipy.interpolate.splev(x_all, tck)

    return y_all


def find_nearest_value(array,value):
    idx = (np.abs(array-value)).argmin()
    return (array[idx])


def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx)


def find_peaks_strain_cwt(period, trace_smooth):

    period = int(period)
    original_len = trace_smooth.shape[0]

    new_smooth = np.ones(period * 2 + original_len) * np.min(trace_smooth)
    new_smooth[period:original_len + period] = trace_smooth

    new_super_smooth = np.ones(period*2 + original_len) * np.min(trace_smooth)
    new_super_smooth[period:original_len+period] = scipy.signal.savgol_filter(trace_smooth, 2*(period//8)+1, 3)


    wavelet_widths = np.arange(period//4, period//2, 1)

    peaks_max_x = np.array(scipy.signal.find_peaks_cwt(new_super_smooth, wavelet_widths, noise_perc=70, min_snr=0.5))
    peaks_max_x = peaks_max_x.astype(np.int64)


    #good_max_peaks = peaks_max_y > np.min(peaks_min_y) + (range * 0.85)
    #good_min_peaks = peaks_min_y < np.min(peaks_min_y) + (range * 0.15)

    #peaks_max_x = peaks_max_x[good_max_peaks]
    #peaks_max_y = peaks_max_y[good_max_peaks]

    #peaks_min_x = peaks_min_x[good_min_peaks]
    #peaks_min_y = peaks_min_y[good_min_peaks]

    peaks_max_rel_x = scipy.signal.argrelextrema(new_smooth, comparator=np.greater, order=period//2)[0]
    peaks_max_rel_x = peaks_max_rel_x.astype(np.int64)

    peaks_max_rel_optim_x = np.zeros_like(peaks_max_x)

    for i, peak_x in enumerate(peaks_max_x):
        peaks_max_rel_optim_x[i] = find_nearest_value(peaks_max_rel_x, peak_x)

    return peaks_max_rel_optim_x - period


def main():

    results = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:

            if not file.endswith(".csv"):
                continue

            print(file)
            file_path = os.path.join(root, file)

            try:
                d = pd.read_csv(file_path)
                period = float(d['period'][0])

                savgol_width = 2 * (period // 12)+1
                savgol_width = max(savgol_width, 5)

                d_lv_len = np.array(d['lv_len'])

                d_hinge = np.array(d['mv_ant_hinge_y'])

                d_filt = filter_lens(d_lv_len)
                d_s_filt = scipy.signal.savgol_filter(d_filt, 11, 3)



                peaks_x_max = find_peaks_strain_cwt(period, d_s_filt)
                peaks_x_min = find_peaks_strain_cwt(period, -d_s_filt)

                #peaks_x_max, _ = scipy.signal.find_peaks(d_s_filt, distance=5, width=5, prominence=0.7)
                #peaks_y_min, _ = scipy.signal.find_peaks(-d_s_filt, distance=5, width=5, prominence=0.7)

                if PLOT:
                    plt.plot(d_lv_len)
                    plt.plot(d_filt)
                    plt.plot(d_s_filt)
                    plt.plot(peaks_x_max, d_s_filt[peaks_x_max], "ro")
                    plt.plot(peaks_x_min, d_s_filt[peaks_x_min], "bo")
                    plt.plot(d_hinge, "g")

                    plt.show()

                out = {
                    "file": file[4:-12],
                    "diastole": d_s_filt[peaks_x_max[0]],
                    "systole": d_s_filt[peaks_x_min[0]]
                }
                print(out)
                results.append(out)

            except Exception as e:
                print(f"{file_path} failed")
                print(e)
                pass

    fieldnames = [
        "file",
        "diastole",
        "systole"
    ]

    with open(CSV_FILE, "w") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for row in results:
            if type(row) is dict:
                writer.writerow(row)


if __name__ == "__main__":
    main()



