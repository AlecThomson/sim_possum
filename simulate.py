#!/usr/bin/env python3
# coding: utf-8

from typing import overload
from astropy.table import Table, hstack
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import multiprocessing as mp
from astropy.modeling import models, fitting
from RMtools_1D import do_RMsynth_1D, do_RMclean_1D
from scipy import interpolate
from functools import partial
from scipy.optimize import curve_fit
import dask
from dask.distributed import Client, LocalCluster
import time
import dask.dataframe as dd

np.random.seed(21092)

# @dask.delayed(nout=2)
def get_cat(cat):
    np.random.seed(21092)
    names = [
        "Source ID",
        "Cluster ID",
        "Galaxy ID",
        "SFtype",
        "AGNtype",
        "Structure parameter (1 = core etc.)",
        "RA",
        "DEC",
        "Comoving distance",
        "redshift",
        "PA",
        "MAJ",
        "MIN",
        "log(I)_151_MHz",
        "log(I)_610_MHz",
        "I(Jy)_1420_MHz",
        "Stokes Q (Jy)",
        "Stokes U (Jy)",
        "Polarized flux density",
        "Fractional polarization",
        "log(I)_4.8_GHz",
        "log(I)_18_GHz",
        "cos(viewing angle)",
        "Rotation Measure",
        "RM flag",
    ]

    skads = Table.read(cat, format="ascii.no_header", names=names)
    n_samp = 10_000
    skads_freqs = np.array([151e6, 610e6, 1420e6, 4.8e9, 18e9])
    snr_idx = skads["I(Jy)_1420_MHz"] > 20e-6 * 5
    # idxs = np.random.randint(low=0, high=len(skads[snr_idx]), size=n_samp)
    # good_cat = skads[snr_idx][idxs]
    good_cat = skads[snr_idx]
    df = good_cat.to_pandas()
    df_dask = dd.from_pandas(df, npartitions=30)
    return df_dask, skads_freqs


@dask.delayed(nout=2)
def get_rm_props(good_cat, sigmas=1):
    np.random.seed(21092)
    phases = np.random.uniform(-np.pi / 2, np.pi / 2, size=len(good_cat))
    sigma_rms = np.random.normal(13, 15, size=len(good_cat)) * sigmas
    sigma_rms[sigma_rms < 0] = 0
    return phases, sigma_rms


np.random.seed(21092)


@dask.delayed(nout=2)
def fitter(
    i, row, freqs, lsq, skads_freqs, phases, sigma_rms, i_noise, q_noise, u_noise
):
    np.random.seed(21092)
    sk_i = [
        10 ** row["log(I)_151_MHz"],
        10 ** row["log(I)_610_MHz"],
        (row["I(Jy)_1420_MHz"]),
        10 ** row["log(I)_4.8_GHz"],
        10 ** row["log(I)_18_GHz"],
    ]
    fitted_line = interpolate.interp1d(
        np.log10(skads_freqs), np.log10(sk_i), kind="quadratic"
    )
    i_spectrum = 10 ** fitted_line(np.log10(freqs))
    qu_spectrum = (
        row["Fractional polarization"]
        * np.exp(2j * (phases[i] + row["Rotation Measure"] * lsq))
        * np.exp(-2 * sigma_rms[i] ** 2 * lsq ** 2)
    )  # (sinc(sigma_rms[i] * lsq))
    qu_spectrum *= i_spectrum
    i_spectrum_noisy = i_spectrum + i_noise[i]
    qu_spectrum_noisy = qu_spectrum + (q_noise[i] + 1j * u_noise[i])
    return i_spectrum_noisy, qu_spectrum_noisy


np.random.seed(21092)


@dask.delayed(nout=2)
def get_band(band):
    np.random.seed(21092)
    data = {
        "band_2": {
            "freqs": np.linspace(1296e6, 1440e6, 144),
            "sens": 28e-6 * np.sqrt(288),
        },
        "band_1": {
            "freqs": np.linspace(800e6, 1087e6, 288),
            "sens": 25e-6 * np.sqrt(288),
        },
        "split_band": {
            "freqs": np.hstack(
                [np.linspace(800e6, 944e6, 144), np.linspace(1296e6, 1440e6, 144)]
            ),
            "sens": 22e-6 * np.sqrt(288),
        },
        "band_1_and_2": {
            "freqs": np.hstack(
                [np.linspace(800e6, 1087e6, 288), np.linspace(1296e6, 1440e6, 144)]
            ),
            "sens": 19e-6 * np.sqrt(288),
        },
    }
    return data[band]["freqs"], data[band]["sens"]


@dask.delayed(nout=3)
def make_noise(good_cat, freqs, sens):
    i_noise = np.random.normal(loc=0, scale=sens, size=(len(good_cat), len(freqs)))
    q_noise = np.random.normal(loc=0, scale=sens, size=(len(good_cat), len(freqs)))
    u_noise = np.random.normal(loc=0, scale=sens, size=(len(good_cat), len(freqs)))
    return i_noise, q_noise, u_noise


@dask.delayed()
def make_data(freqs, i_spectrum_noisy, qu_spectrum_noisy, sens):
    data = [
        freqs,
        i_spectrum_noisy,
        np.real(qu_spectrum_noisy),
        np.imag(qu_spectrum_noisy),
        np.ones_like(i_spectrum_noisy) * sens,
        np.ones_like(i_spectrum_noisy) * sens,
        np.ones_like(i_spectrum_noisy) * sens,
    ]
    return data


@dask.delayed(nout=3)
def get_vals(mDicts, aDicts):
    fdf_mads = [m["dFDFcorMAD"] for m in mDicts]
    snrs = [m["snrPIfit"] for m in mDicts]
    rms = [m["phiPeakPIfit_rm2"] for m in mDicts]
    return fdf_mads, snrs, rms


@dask.delayed()
def make_cat(good_cat, mDicts, mDict_cls, pas, sigmas):
    good_tab = Table.from_pandas(good_cat)
    syn_tab = Table({k: [dic[k] for dic in mDicts] for k in mDicts[0]})
    clean_tab = Table({k: [dic[k] for dic in mDict_cls] for k in mDict_cls[0]})
    good_tab.add_column(pas, name='Pol_Angle')
    good_tab.add_column(sigmas, name='Sigma_RM')
    tab = hstack([good_tab,syn_tab,clean_tab], table_names=['INPUT','SYNTH','CLEAN'])
    return tab


rmsynth = dask.delayed(do_RMsynth_1D.run_rmsynth, nout=2)
rmclean = dask.delayed(do_RMclean_1D.run_rmclean, nout=2)


def main():
    np.random.seed(21092)
    cluster = LocalCluster(n_workers=31, threads_per_worker=1)
    # cluster = LocalCluster()
    print(cluster)
    client = Client(cluster)
    print(client)
    good_cat, skads_freqs = get_cat(
        "/scratch/tho822/master_possum_catalogue_trim10x10deg.dat"
    )
    phases, sigma_rms = get_rm_props(good_cat)
    for band in ["band_2", "band_1", "split_band", "band_1_and_2"]:
    # for band in ["band_2"]:
        freqs, sens = get_band(band)
        freqs = freqs.compute()
        lsq = (2.998e8 / np.asarray(freqs)) ** 2
        i_noise, q_noise, u_noise = make_noise(good_cat, freqs, sens)

        mDicts, aDicts = [], []
        mDict_cls, aDict_cls = [], []
        for i, row in good_cat.iterrows():
            i_spectrum_noisy, qu_spectrum_noisy = fitter(
                i, row, freqs, lsq, skads_freqs, phases, sigma_rms, i_noise, q_noise, u_noise
            )
            data = make_data(freqs, i_spectrum_noisy, qu_spectrum_noisy, sens)
            mDict, aDict = rmsynth(
                data=data,
                dPhi_radm2=10,
                phiMax_radm2=2000,
                polyOrd=-3,
                fit_function="log",
            )
            mDicts.append(mDict)
            aDicts.append(aDict)
            mDict_cl, aDict_cl = rmclean(mDict, aDict, cutoff=-5)
            mDict_cls.append(mDict_cl)
            aDict_cls.append(aDict_cl)
        tab = make_cat(good_cat, mDicts, mDict_cls, phases, sigma_rms)
        # time.sleep(10)
        tab = tab.compute()
        tab.write(f"{band}_sim_cat.csv", format='pandas.csv')


if __name__ == "__main__":
    main()
