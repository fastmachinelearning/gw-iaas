import os
from typing import Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


def get_asd(ts: np.ndarray) -> FrequencySeries:
    ts = TimeSeries(ts, sample_rate=4096.0)
    asd = ts.asd(fftlength=10, overlap=0.5, method="median")
    return asd.crop(20, 300)


def downsample(fs, target):
    """Map frequency series `fs` onto the frequencies of `target`"""
    interped = np.interp(target.frequencies, fs.frequencies, fs.value)
    return FrequencySeries(interped, f0=target.frequencies[0], df=target.df)


def make_plots(
    results: Mapping[str, np.ndarray],
    ratio: Tuple[str, str],
    freq_lo: float = 55,
    freq_hi: float = 65,
) -> plt.Figure:
    fig = plt.figure(figsize=(13, 5))

    # plot raw ASDs on the first plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("ASD")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"ASD [strain / ${\sqrt{Hz}}$]")
    for label in results.keys():
        ts = results[label]

        # map the timeseries to a frequency series if need be
        if not isinstance(ts, FrequencySeries):
            fs = get_asd(ts)
            results[label] = fs

        ax1.loglog(results[label], label=label)
    ax1.legend()

    # plot the ASD ratio of two of the ASD
    # curves on the second plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("ASD ratio ({} / {})".format(*ratio))
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("ASD ratio")

    # interpolate everything down to the lowest
    # resolution if their resolutions don't match
    asds = [results[ratio[i]] for i in range(2)]
    if asds[0].df > asds[1].df:
        asds[1] = downsample(asds[1], asds[0])
    elif asds[1].df > asds[0].df:
        asds[0] = downsample(asds[0], asds[1])

    ratio = asds[0] / asds[1]
    ax2.loglog(ratio.crop(freq_lo, freq_hi))
    return fig


def plot_results(raw_dir: str, clean_dir: str, channel_name: str, fname: str):
    clean_ts = TimeSeries()
    raw_ts = TimeSeries()
    for f in os.listdir(clean_dir):
        ts = TimeSeries.read(os.path.join(clean_dir, f), channel_name)
        clean_ts.append(ts)

        ts = TimeSeries.read(os.path.join(raw_dir, f), channel_name)
        raw_ts.append(ts)

    fig = make_plots(
        {"Raw": raw_ts, "Cleaned": clean_ts}, ratio=("Raw", "Cleaned")
    )
    fig.savefig(fname)
