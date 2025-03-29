import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional

from . import data_loading


def get_calibration_parameters(data_dir: str|None = None, num_files: int = -1, files: list[str|Path]|None = None, 
                               far_probe_avrg_tol: int = 100,  folders: Optional[list[str]] = None, 
                               verbose: bool = False) -> dict[str, float]:
    """
    Get calibration parameters from the calibration files found in the data-directory.

    Keyword Arguments:
        data_dir {str} -- Path to the data-directory. Not used if `files` is provided. (default: {None})
        files {list[str|Path]} -- List of calibration filepaths. If None, the files are loaded from the data-directory. 
                                  (default: {None})
        num_files {int} -- Number of files to load. -1 or 0 to load all files. Not used if `files` is provided.
                           (default: {-1})
        far_probe_avrg_tol {int} -- The number of measuring steps at the beginning/end of the approach/retract curve 
                                    over which the measurements are averaged for the parameters of the probe at maximum distance.
                                    (default: {100})
        folders {list[str]} -- List of sub-folders in the data-directory to get calibration files from. 
                               If None, all folders are selected. Not used if `files` is provided. (default: {None})
        verbose {bool} -- Print calibration parameters. (default: {False})

    Returns:
        {dict[str, float]}: Dictionary containing the following calibration parameters:
            - phase_far: Phase far
            - amp_far: Amplitude far
            - error_phase: Error in phase
            - error_amp: Error in amplitude
            - kc: Spring constant
            - Qfact: Quality factor
            - freq_thermal: Thermal frequency
            - freq_drive: Drive frequency
    """
    if files is None:
        print("Calibration: Read ibw-files and get calibration parameters ...")
        files = data_loading.get_ibw_paths(data_dir, calib_only=True, n=num_files, folders=folders)

    param_data = []
    for cf in files:
        ibw_data = data_loading.load_ibw_force(cf)
        params = matz_AMcalibration(ibw_data, far_probe_avrg_tol)
        param_data.append(params)
    
    df = pd.DataFrame(param_data)
    drives = df['drive'].values
    std_phase = df['phase_far'].std()
    std_amp = df['amp_far'].std()
    df = df.drop(columns=['name', 'drive'])
    
    max_drive_len = max([len(d) for d in drives])
    drives = np.array([np.pad(d, (0, max_drive_len - len(d)), constant_values=np.nan) for d in drives])
    drive = np.nanmean(drives, axis=1)
    
    calib_params = df.mean().to_dict()
    calib_params['drive'] = drive
    calib_params['std_phase'] = std_phase
    calib_params['std_amp'] = std_amp

    if verbose:
        print("Calibration parameters:")
        for k, v in calib_params.items():
            print(f"   {k}: {v}")
    return calib_params


def matz_AMcalibration(ibw_data: tuple[dict[str, str], list[str], np.ndarray, str],
                       far_probe_avrg_tol: int = 100) -> dict[str, float]:
    constants, labels, wave_data, name = ibw_data
    ##################### DATA PLAIN TEXT #####################
    # Read out the plain text data: Spring constant and InvOLS 
    # Further data can be added. Just look at ibw file with a text editor and choose what you need.
    params = {'kc': float(constants['SpringConstant']),                              # spring constant..............................k [nN/nm]
              'InvOLS': float(constants['InvOLS']) * 10**9,                          # inverse optical lever Sensitivity............s [m/V]
              'AmpInvOLS': float(constants['AmpInvOLS']) * 10**9,                    # AmpInvOLS in.................................[m/V]
              'DriveFrequency': float(constants['DriveFrequency']),                  # Excitation frequency or Drive Frequency......[Hz]
              'ThermalFrequency': float(constants['ThermalFrequency']),              # EigenFrequency or Resonance Frequency........[Hz]
              'FreeAirAmplitude': float(constants['FreeAirAmplitude']),              # Tuned amplitude after Tuning.................[V]
              'FreeAirPhase': float(constants['FreeAirPhase']),                      # Phase set a certain distance above surface...(deg)
              'AmplitudeSetpointVolts': float(constants['AmplitudeSetpointVolts']),  # Setpoint amplitude for surface approach......[V]
              'Qfact': float(constants['ThermalQ']),                                 # quality factor...............................(dimensionless)
              'name': name}

    ##################### DATA COMPUTED #####################
    phase = wave_data[:, labels.index('Phase')]              # deg
    amp = wave_data[:, labels.index('Amp')]                  # m
    drive = wave_data[:,labels.index('Drive')]               # m
    params['drive'] = np.abs(drive - np.max(drive)) * 10**9  # nm

    pf_app = np.mean(phase[:far_probe_avrg_tol])
    sd_pf_app = (np.std(phase[:far_probe_avrg_tol])**2) / len(phase[:far_probe_avrg_tol])
    pf_ret = np.mean(phase[-far_probe_avrg_tol:])
    sd_pf_ret = (np.std(phase[-far_probe_avrg_tol:])**2) / len(phase[-far_probe_avrg_tol:])
    phase_far = np.array([pf_app, pf_ret])
    params['phase_far'] = np.mean(phase_far)
    params['error_phase'] = np.sqrt(sd_pf_app + sd_pf_ret)

    af_app = np.mean(amp[:far_probe_avrg_tol])
    sd_af_app = (np.std(amp[:far_probe_avrg_tol])**2) / len(amp[:far_probe_avrg_tol])
    af_ret = np.mean(amp[-far_probe_avrg_tol:])
    sd_af_ret = (np.std(amp[-far_probe_avrg_tol:])**2) / len(amp[-far_probe_avrg_tol:])
    amp_far = np.array([af_app, af_ret])
    params['amp_far'] = np.mean(amp_far)
    params['error_amp'] = np.sqrt(sd_af_app+sd_af_ret)

    return params
