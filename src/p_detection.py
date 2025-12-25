import wfdb
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample, find_peaks, medfilt, medfilt,  savgol_filter
import pandas as pd
import h5py
from math import nan
from xlsxwriter import Workbook
from typing import List, Optional, Tuple, Union, Dict


def filter_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply baseline wander removal and smoothing to an ECG signal using
    two-stage median filtering followed by Savitzky-Golay smoothing.

    The process removes low-frequency drift (baseline wander) and smooths
    high-frequency noise while preserving the signal morphology.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal to be filtered.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    """
    if fs <= 1:
        raise ValueError("Sampling frequency must be greater than 1 Hz.")
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1D NumPy array.")

    median_filtered_signal1 = medfilt(signal, kernel_size=fs-1)
    median_filtered_signal = medfilt(
        median_filtered_signal1, kernel_size=2*fs-1)
    ecg_deducted = signal-median_filtered_signal
    filtered_ecg = savgol_filter(ecg_deducted, window_length=30, polyorder=4)

    return filtered_ecg


def compute_rr_intervals(r_peaks: list | np.ndarray, fs: int) -> np.ndarray:
    """
    Compute RR intervals (in seconds) from consecutive R-peak indices in an ECG signal.

    The RR interval is the time between successive R-peaks and is critical for analyzing
    heart rate variability and detecting arrhythmias.

    Parameters
    ----------
    r_peaks : list or np.ndarray
        Indices of R-peaks (in samples).
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of RR intervals in seconds.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
    return rr_intervals


def phasor_transform(ecg_signal: np.ndarray, rv: float) -> np.ndarray:
    """
    Apply the phasor transformation to an ECG signal segment using a reference value.

    This transformation highlights phase changes and is especially useful in identifying
    peaks and inflection points in periodic biological signals like ECG.

    Formula
    -------
    PT[n] = arctangent(ecg_signal[n] / rv)

    Parameters
    ----------
    ecg_signal : np.ndarray
        ECG signal segment (1D array).
    rv : float
        Reference voltage or scaling constant used to normalize the signal.

    Returns
    -------
    np.ndarray
        Transformed signal using the phasor method.

    Raises
    ------
    ValueError
        If rv is zero or if input is not 1D.
    """
    if rv == 0:
        raise ValueError("Reference voltage (rv) must not be zero.")
    if ecg_signal.ndim != 1:
        raise ValueError("Input ECG signal must be a 1D array.")

    return np.arctan(ecg_signal / rv)  # PT(n) = tan⁻¹(x(n) / RV)


def pathology_check(
    r_peaks: List[int],
    afib_labels: List[bool],
    pvc_labels: List[bool],
    pause_labels: List[bool],
) -> List[Optional[int]]:
    """
    Identify valid R-peaks for P-wave detection by excluding pathological beats.

    This function filters out R-peaks associated with arrhythmic conditions such as
    atrial fibrillation (AFIB), premature ventricular contractions (PVC), or pauses.
    These are known to distort or mask P-waves and are marked as invalid (None).

    Parameters
    ----------
    r_peaks : List[int]
        List of R-peak indices.
    afib_labels : List[bool]
        List of AFIB flags corresponding to R-peaks.
    pvc_labels : List[bool]
        List of PVC flags corresponding to R-peaks.
    pause_labels : List[bool]
        List of pause flags corresponding to R-peaks.

    Returns
    -------
    List[Optional[int]]
        List of valid R-peak indices or None for pathological beats.

    Raises
    ------
    ValueError
        If input lists are not of the same length.
    """
    if not (len(r_peaks) == len(afib_labels) == len(pvc_labels) == len(pause_labels)):
        raise ValueError("All input lists must have the same length.")

    valid_beats = []
    for i in range(len(r_peaks)):
        # Skip AFIB and PVC beats  \\\afib_labels[i] or pvc_labels[i] or
        if afib_labels[i] or pvc_labels[i] or pause_labels[i]:
            valid_beats.append(None)
        else:
            valid_beats.append(r_peaks[i])

    return valid_beats  # Return logic indices where P-wave detection should proceed


def rr_for_p(
    p_peaks: Union[List[int], np.ndarray],
    r_peaks: Union[List[int], np.ndarray],
) -> List[int]:
    """
    Identify the RR interval index in which each P-wave peak falls.

    This function checks for each P-wave peak where it lies between two consecutive
    R-peaks and returns the corresponding RR interval index.

    Parameters
    ----------
    p_peaks : list or np.ndarray
        Detected P-peak sample indices.
    r_peaks : list or np.ndarray
        Detected R-peak sample indices.

    Returns
    -------
    List[int]
        A list of indices such that each element `i` indicates the RR interval
        between r_peaks[i] and r_peaks[i+1] where the corresponding P-peak falls.
        If a P-peak does not fall in any RR interval, -1 is returned.

    """
    rr_indices_for_p = []

    for p in p_peaks:
        for i in range(len(r_peaks) - 1):
            if r_peaks[i] < p < r_peaks[i + 1]:
                # P-peak is in RR interval starting at r_peaks[i]
                rr_indices_for_p.append(i)
                break
        else:
            # if p does not fall within any RR interval, append -1
            rr_indices_for_p.append(-1)

    return rr_indices_for_p


def extract_segments(
    signal: np.ndarray,
    rr_int: Union[List[float], np.ndarray],
    r_ind: List[int],
    fs: int,
    valid_beats: List[Optional[int]],
    begin: int,
    dis_p_check: bool = False,
    det_t: Optional[List[int]] = None,
    det_p: Optional[List[int]] = None
) -> Union[
    Tuple[List[List[float]], List[int]],
    Tuple[List[List[float]], List[List[float]], List[List[float]], List[float]]
]:
    """
    Extract ECG segments based on valid R-peaks and RR intervals.

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal segment.
    rr_int : list or np.ndarray
        List of RR intervals in seconds.
    r_ind : list
        Indices of detected R-peaks.
    fs : int
        Sampling frequency in Hz.
    valid_beats : list
        R-peaks considered valid for segmentation (None if invalid).
    begin : int
        Index offset to match full signal context.
    dis_p_check : bool, optional
        If True, extract dissociated segments instead.
    det_t : list, optional
        Detected T-wave peaks.
    det_p : list, optional
        Detected P-wave peaks.

    Returns
    -------
    If dis_p_check is False:
        segments : list of list
            Full extracted RR segments.
        segments_p : list of list
            Segments in expected P-wave windows.
        segments_t : list of list
            Segments in expected T-wave windows.
        win_val : list
            Placeholder for window values (currently unused).
    If dis_p_check is True:
        segements_dis : list of list
            Segments between T and P peaks to analyze dissociated P-waves.
        t_val : list
            T-peak matching result per RR interval.
    """

    segments = []
    segments_p = []
    segments_t = []
    segements_dis = []
    count = 0
    win_val = []
    p_val = []
    t_val = []
    if det_p is not None and det_t is not None:
        for i in range(len(rr_int)):
            p_match = next(
                (p for p in det_p if r_ind[i] < p < r_ind[i + 1]), None)
            t_match = next(
                (t for t in det_t if r_ind[i] < t < r_ind[i + 1]), None)

            p_val.append(p_match)
            t_val.append(t_match)

    for i in range(len(rr_int)):
        # Check if the current beat is valid using valid_beats

        if dis_p_check == False:
            if valid_beats[i+1] is not None:
                start_t = r_ind[i] + int(0.12 * rr_int[i] * fs)
                if rr_int[i] < 1.5:
                    end_t = r_ind[i] + \
                        int((0.57 * rr_int[i] + 0.040) * fs)
                else:
                    end_t = r_ind[i] + \
                        int((0.3 * rr_int[i] + 0.060) * fs)
                if rr_int[i] < 0.54:
                    start_p = r_ind[i] + \
                        int(0.5 * rr_int[i] * fs)
                    end_p = r_ind[i + 1] - \
                        int((0.07 * rr_int[i] + 0.07) * fs)
                elif 0.54 <= rr_int[i] < 0.8:
                    start_p = r_ind[i] + \
                        int(0.5 * rr_int[i] * fs)
                    end_p = r_ind[i + 1] - \
                        int((
                            0.07 * rr_int[i] + 0.05) * fs)+5
                elif 0.8 <= rr_int[i] < 1.2:
                    start_p = r_ind[i] + \
                        int(0.6 * rr_int[i] * fs)-5
                    if signal[r_ind[i+1]-begin] < 0:
                        end_p = r_ind[i + 1] - \
                            int((
                                0.07 * rr_int[i] + 0.04) * fs)
                    else:
                        end_p = r_ind[i + 1] - \
                            int((
                                0.07 * rr_int[i] + 0.01) * fs)
                elif 1.2 <= rr_int[i] < 1.78:
                    start_p = r_ind[i] + \
                        int(0.7 * rr_int[i] * fs)+30
                    if rr_int[i] >= 1.5:
                        end_p = r_ind[i + 1] - \
                            int((0.07 * rr_int[i]) * fs)+20
                    else:
                        end_p = r_ind[i + 1] - \
                            int((0.07 * rr_int[i]) * fs)

                elif rr_int[i] >= 1.78:
                    start_p = r_ind[i] + \
                        int(0.7 * rr_int[i] * fs)+30
                    end_p = r_ind[i + 1] - \
                        int((0.07 * rr_int[i]) * fs)+25

                else:
                    start_p = r_ind[i] + \
                        int(0.6 * rr_int[i] * fs)-5
                    if signal[r_ind[i+1]-begin] < 0:
                        end_p = r_ind[i + 1] - \
                            int((
                                0.07 * rr_int[i] + 0.04) * fs)
                    else:
                        end_p = r_ind[i + 1] - \
                            int((
                                0.07 * rr_int[i] + 0.01) * fs)

                # Ensure the end index does not exceed signal length
                end = min(end_p, len(signal)+begin)
                # Extract segment
                if rr_int[i] < 3:
                    a = signal[start_t-begin:end_p-begin]
                    if (end_p-start_p) >= 20:
                        segments_p.append(
                            np.array(signal[start_p-begin:end_p-begin]).tolist())
                        segments_t.append(
                            np.array(a[:end_t-start_t]).tolist())
                        segments.append(np.array(a).tolist())
                    else:
                        segments.append([])
                        segments_p.append([])
                        segments_t.append([])
                else:
                    segments.append([])
                    segments_p.append([])
                    segments_t.append([])

            else:  # Added else block to append empty list if beat is invalid
                segments.append([])
                segments_p.append([])
                segments_t.append([])

        elif dis_p_check:
            if len(valid_beats) == 0:
                segements_dis.append([])
            elif len(valid_beats) != 0 and valid_beats[i]:
                if len(p_val) == 0 or len(t_val) == 0:
                    segements_dis.append([])

                elif p_val[i] is not None and t_val[i] is not None:
                    start_dis = t_val[i] + int(0.2*fs)
                    end_dis = p_val[i] - int(0.4*fs)
                    end = min(end_dis, len(signal)+begin)
                    if (end-start_dis) >= 70:
                        segements_dis.append(
                            np.array(signal[start_dis-begin:end-begin]).tolist())

                    else:
                        segements_dis.append([])
                        t_val[i] = None
                        p_val[i] = None

            else:
                segements_dis.append([])
        else:
            segments.append([])
            segments_p.append([])
            segments_t.append([])
            segements_dis.append([])

    if dis_p_check:
        return segements_dis, t_val  # returns only 2 values
    else:
        return segments, segments_p, segments_t, win_val  # returns 4 values


def generate_template(segments, pearson_threshold: float = 0.8):
    """
    Cluster ECG segments into templates based on Pearson correlation similarity.

    Each segment is resampled and compared against existing templates. If similarity exceeds
    the given Pearson threshold, the segment is averaged into the template; otherwise,
    a new template is created.

    Parameters
    ----------
    segments : list of list
        ECG segments to cluster.
    pearson_threshold : float
        Correlation threshold to match existing template.

    Returns
    -------
    templates : list of np.ndarray
        Final list of templates.
    counts : list of int
        Number of segments assigned to each template.
    all_scores : list of float
        Pearson scores computed during assignment.
    check_list : list of np.ndarray
        Copies of original seed segment for each template.
    segment_to_template : dict
        Mapping from segment index to template index.
    """

    segments_new = segments.copy()

    if len(segments_new) == 0:
        # print("Not enough segments to generate templates.")
        return [], [], [], [], {}

    templates = []
    counts = []
    all_scores = []
    check_list = []
    segment_to_template = {}
    first_segment_assigned = False

    # Step 1: Assign first valid segment as first template
    for seg_idx, seg in enumerate(segments_new):
        if seg:
            first_segment = np.array(seg, dtype=float).flatten()
            templates.append(first_segment)
            counts.append(1)
            check_list.append(first_segment)
            segment_to_template[seg_idx] = 0
            first_segment_assigned = True
            break

    # Step 2: Process remaining segments
    for seg_idx, seg in enumerate(segments_new):
        if not seg or (first_segment_assigned and seg_idx == 0):
            continue

        seg = np.array(seg, dtype=float).flatten()
        matched = False

        for temp_idx, temp in enumerate(templates):
            # Resample to match lengths
            target_len = 50  # Desired length
            # Flatten both inputs
            seg = seg.flatten()
            temp = temp.flatten()

            if len(seg) > len(temp):
                if len(temp) < target_len:
                    # Cubic interpolation on temp
                    x_old = np.linspace(0, 1, len(temp))
                    x_new = np.linspace(0, 1, target_len)
                    cubic_interp = interp1d(x_old, temp, kind='cubic')
                    temp_resampled = cubic_interp(x_new)

                    # Resample seg to target_len using FFT-based resampling
                    seg_resampled = resample(
                        seg, target_len).flatten()  # type: ignore
                else:
                    seg_resampled = resample(
                        seg, len(temp)).flatten()  # type: ignore
                    temp_resampled = temp
            else:
                if len(seg) < target_len:
                    # Cubic interpolation on seg
                    x_old = np.linspace(0, 1, len(seg))
                    x_new = np.linspace(0, 1, target_len)
                    cubic_interp = interp1d(x_old, seg, kind='cubic')
                    seg_resampled = cubic_interp(x_new)

                    # Resample temp to target_len using FFT-based resampling
                    temp_resampled = resample(
                        temp, target_len).flatten()  # type: ignore
                else:
                    temp_resampled = resample(
                        temp, len(seg)).flatten()  # type: ignore
                    seg_resampled = seg

                # Compute Pearson correlation
            df = pd.DataFrame({'x': temp_resampled, 'y': seg_resampled})
            pearson_corr = df.corr(method='pearson').iloc[0, 1]

            all_scores.append((pearson_corr))
            if pearson_corr >= pearson_threshold:  # type: ignore
                templates[temp_idx] = (
                    temp_resampled * counts[temp_idx] + seg_resampled) / (counts[temp_idx] + 1)
                counts[temp_idx] += 1
                segment_to_template[seg_idx] = temp_idx
                matched = True
                break

        if not matched:
            templates.append(seg)
            counts.append(1)
            check_list.append(seg)
            segment_to_template[seg_idx] = len(templates) - 1
    return templates, counts, all_scores, check_list, segment_to_template


def drop_low_weight_templates(
    segs: List[List[float]],
    templates: List[np.ndarray],
    counts: List[int],
    segment_to_template: Dict[int, int],
    qrs_indices: List[int],
    perce: float
) -> Tuple[
    List[np.ndarray], List[int], Dict[int, int], List[List[float]]
]:
    """
    Drop templates with low segment count or minimal contribution to QRS detections.

    Parameters
    ----------
    segs : list of list
        Original list of ECG segments.
    templates : list of np.ndarray
        Template signals.
    counts : list of int
        Segment counts per template.
    segment_to_template : dict
        Mapping from segment index to template index.
    qrs_indices : list of int
        Detected QRS indices.
    perce : float
        Drop threshold as a percentage of QRS coverage.

    Returns
    -------
    updated_templates : list of np.ndarray
    updated_counts : list of int
    updated_segment_to_template : dict
    updated_segments : list of list
    """
    total_qrs = len(qrs_indices)
    drop_threshold = perce * total_qrs  # perce% of total QRS indices
    removed_count = 0  # Track the total count removed
    drop_indices = []  # Store templates to be removed

    # Sort templates by count (ascending) and track their indices
    sorted_templates = sorted(
        enumerate(counts), key=lambda x: x[1])  # (index, count)

    for idx, count in sorted_templates:
        # Skip templates with sufficient counts or skip this for very short or long steady signals
        if count >= 10 or len(sorted_templates) <= 5:
            continue
        if count == 1:
            drop_indices.append(idx)
            if len(sorted_templates) <= 50:
                removed_count += count
            else:
                continue
        else:
            removed_count += count

        if removed_count + count > drop_threshold:
            break  # Stop if 5% threshold is reached

        drop_indices.append(idx)

    # print(f"Dropping templates: {drop_indices} (Total Removed: {removed_count})")

    # Create updated lists
    updated_templates = [t for i, t in enumerate(
        templates) if i not in drop_indices]
    updated_counts = [c for i, c in enumerate(
        counts) if i not in drop_indices]

    # Create mapping from old index to new index (since templates are removed)
    template_index_map = {}
    new_idx = 0
    for old_idx in range(len(templates)):
        if old_idx not in drop_indices:
            template_index_map[old_idx] = new_idx
            new_idx += 1

    # Assign [] to segments belonging to removed templates
    updated_segment_to_template = {}
    updated_segments = segs.copy()  # This will contain the updated segment list

    for seg_idx, temp_idx in segment_to_template.items():
        if temp_idx in drop_indices:
            # []  # Mark as unassigned
            updated_segment_to_template[seg_idx] = None
            # Ensure the segment array is updated
            updated_segments[seg_idx] = []
        else:
            updated_segment_to_template[seg_idx] = template_index_map[temp_idx]
            # Store updated assignment

    return updated_templates, updated_counts, updated_segment_to_template, updated_segments


def safe_savgol_filter(
    x: np.ndarray,
    window_length: int = 10,
    polyorder: int = 3
) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter safely, adapting to short signals.

    If the signal is too short for the given window or polynomial order,
    the function reduces the window or skips smoothing.

    Parameters
    ----------
    x : np.ndarray
        Signal to smooth.
    window_length : int
        Desired filter window length (must be odd).
    polyorder : int
        Polynomial order for smoothing.

    Returns
    -------
    np.ndarray
        Smoothed signal or original signal if too short.
    """
    if len(x) < window_length:
        # Make window_length the largest odd number <= len(x)
        window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        if window_length < polyorder + 2:
            # Not enough points for given polyorder — return original or smoothed another way
            return np.array(x)
    return savgol_filter(x, window_length=window_length, polyorder=polyorder)


def amp(signal: np.ndarray, peak: int, window_size: int) -> float:
    """
    Calculates a symmetric local amplitude around a peak.

    Parameters:
    - signal (np.array): The input signal.
    - peak (int): Index of the peak in the signal.
    - window_size (int): Number of samples to look on either side.

    Returns:
    - float: The minimum of the left and right segment peak amplitudes.
    """
    left_idx = max(0, peak - window_size)
    right_idx = min(len(signal), peak + window_size + 1)

    l_seg = signal[left_idx:peak + 1]
    r_seg = signal[peak:right_idx]
    l_seg = l_seg - np.min(l_seg)
    r_seg = r_seg - np.min(r_seg)

    l_amp = np.max(l_seg) if len(l_seg) > 0 else 0
    r_amp = np.max(r_seg) if len(r_seg) > 0 else 0

    return min(l_amp, r_amp)


def find_best_peak(
    new: np.ndarray,
    new_minus: np.ndarray,
    tem: np.ndarray,
    tem_min: np.ndarray,
    template: np.ndarray,
    plot: bool = False,
    num: int = 0,
    dis_p: bool = False
) -> Optional[int]:
    """
    Find the most prominent peak in phasor-transformed ECG signal based on
    amplitude and width, using both positive and negative phasor components.

    Parameters
    ----------
    new : np.ndarray
        Positive phasor-transformed signal.
    new_minus : np.ndarray
        Negative phasor-transformed signal (mirrored).
    tem : np.ndarray
        Shifted positive signal for scoring.
    tem_min : np.ndarray
        Shifted negative signal for scoring.
    template : np.ndarray
        Original ECG template for plotting.
    plot : bool, optional
        Whether to visualize peak selection (default is False).
    num : int, optional
        Template index for plotting title (default is 0).
    dis_p : bool, optional
        If True, force use of positive peaks only (for dissociated P-wave detection).

    Returns
    -------
    selected_peak : int or None
        Index of the best peak, or None if no valid peak is found.
    """
    # Shift signals to be non-negative
    """
    new_shifted = new - np.min(new) if np.any(new < 0) else new.copy()
    new_shifted_min = new_minus - \
        np.min(new_minus) if np.any(new_minus < 0) else new_minus.copy()"""

    # Find candidate peaks and their properties
    peaks_tem, prop_tem = find_peaks(new, prominence=0.05, width=0)
    peaks_min, prop_min = find_peaks(new_minus, prominence=0.1, width=0)
    new_shifted = tem
    new_shifted_min = tem_min
    if dis_p:
        if len(peaks_tem) > 0:
            scores = [
                amp(new_shifted, p, 10) * w
                for p, w in zip(peaks_tem, prop_tem["widths"])
            ]

            # peaks_tem[np.argmax(scores)]
            selected_peak = np.argmin(new_minus)
        else:
            selected_peak = None
    else:
        if len(peaks_tem) == 0 and len(peaks_min) == 0:
            if plot:
                plt.figure(figsize=(12, 5))
                plt.plot(template, label='Original Template', linewidth=1.5)
                plt.plot(np.arange(len(new)), new,
                         label='Phasor ', linestyle='-.')
                plt.plot(np.arange(len(new_minus)), new_minus,
                         label='minus Phasor', linestyle='-.')
                plt.title(f'Template {num} with No Detected Peaks')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            return None

        elif len(peaks_tem) == 0 and len(peaks_min) > 0:
            scores = [
                amp(new_shifted_min, p, 10) * w
                for p, w in zip(peaks_min, prop_min["widths"])
            ]
            selected_peak = peaks_min[np.argmax(scores)]

        elif len(peaks_tem) > 0 and len(peaks_min) == 0:
            scores = [
                amp(new_shifted, p, 10) * w
                for p, w in zip(peaks_tem, prop_tem["widths"])
            ]
            selected_peak = peaks_tem[np.argmax(scores)]

        else:
            scores_tem = [
                amp(new_shifted, p, 10) * w
                for p, w in zip(peaks_tem, prop_tem["widths"])
            ]
            scores_min = [
                amp(new_shifted_min, p, 10) * w
                for p, w in zip(peaks_min, prop_min["widths"])
            ]

            peak_pos = peaks_tem[np.argmax(scores_tem)]
            peak_min = peaks_min[np.argmax(scores_min)]

            if amp(new_shifted_min, peak_min, 10) > 3 * amp(new_shifted, peak_pos, 10):
                selected_peak = peak_min
            else:
                selected_peak = peak_pos

    # Plotting
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(template, label='Original Template', linewidth=1.5)
        if selected_peak is not None:
            plt.axvline(x=selected_peak, color='black',  # type: ignore
                        linestyle='-', linewidth=2, label='Selected Peak')
        plt.plot(np.arange(len(new)), new,
                 label='Shifted Phasor', linestyle='-.')
        plt.plot(np.arange(len(new_minus)), new_minus,
                 label='Minus Phasor', linestyle='-.')
        plt.title(f'Template {num} with Candidate and Selected Peaks')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_peak  # type: ignore


def detect_peaks_in_templates(
    templates: List[np.ndarray],
    plot: bool = False,
    pt_v: float = 0.01,
    dis_p: bool = False
) -> List[Optional[int]]:
    """
    Detect peak positions within ECG templates using phasor transformation and
    peak scoring logic.

    Parameters
    ----------
    templates : list of np.ndarray
        List of ECG waveform templates.
    plot : bool, optional
        If True, generate plots for each detection.
    pt_v : float, optional
        Scaling constant for the phasor transformation.
    dis_p : bool, optional
        If True, detect only dissociated P-wave peaks.

    Returns
    -------
    List[Optional[int]]
        List of detected peak indices (or None if no peak found).
    """
    peak_indices = []

    for i, template in enumerate(templates):
        # print(f"\n--- Template {i+1} ---"
        if np.any(template < 0):
            # shift the signal so the min becomes 0
            tem_shifted = template - np.min(template)
        else:
            tem_shifted = template.copy()  # no need to shift
        tem_min = -np.array(template)
        if np.any(tem_min < 0):
            # shift the signal so the min becomes 0
            tem_shifted_min = tem_min - np.min(tem_min)
        else:
            tem_shifted_min = tem_min.copy()

        phasor_trans = (phasor_transform(tem_shifted, pt_v))
        pt_minus = (phasor_transform(tem_shifted_min, pt_v))
        new_tem = template

        final_peak = find_best_peak(
            phasor_trans, pt_minus, tem_shifted, tem_shifted_min, template, plot=plot, num=i, dis_p=dis_p)

        if final_peak is not None:
            final_peak += 0

        peak_indices.append(final_peak)

    return peak_indices


def map_peaks_to_real_segments(
    signal: np.ndarray,
    templates: List[np.ndarray],
    segments_all: List[np.ndarray],
    segments: List[np.ndarray],
    segment_to_template: dict,
    peak_indices: List[Optional[int]],
    p: bool,
    dis: bool,
    qrs_indices: List[int],
    rr_int: List[float],
    begin: int,
    search_window: int = 8,
    det_t: Optional[List[int]] = None,
    dis_p: Optional[List[bool]] = None,
    fs: Optional[int] = 250
) -> List[int]:
    """
    Maps detected peaks from templates back to the original ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Full ECG signal.
    templates : list of np.ndarray
        Templates used for detection.
    segments_all : list of np.ndarray
        All original full-length segments.
    segments : list of np.ndarray
        Aligned segments for each template.
    segment_to_template : dict
        Maps segment index to template index.
    peak_indices : list
        Peak index in each template.
    p : bool
        Whether this is for P-wave mapping.
    dis : bool
        Whether this is for dissociated P-wave mapping.
    qrs_indices : list of int
        R-peaks (per segment).
    rr_int : list of float
        RR intervals (in seconds).
    begin : int
        Start index of current signal window.
    search_window : int
        Number of samples around mapped peak to search in original.
    det_t : list of int, optional
        T-wave peaks (for dis mapping).
    dis_p : list of bool, optional
        Dissociated flags (per segment).

    Returns
    -------
    list of int
        Indices of mapped peaks in the original signal.
    """
    real_peak_indices = {}
    real_ind = []

    for seg_idx, seg in enumerate(segments):
        temp_idx = segment_to_template.get(seg_idx)

        if seg is not None and temp_idx is not None and peak_indices[temp_idx] is not None:
            template_peak_ratio = peak_indices[temp_idx] / \
                len(templates[temp_idx])

            # Map the peak to the original segment length
            mapped_idx = int(template_peak_ratio * len(seg))

            # Define search window bounds
            start_idx = max(mapped_idx - search_window, 0)
            end_idx = min(mapped_idx + search_window, len(seg) - 1)

            # Find the actual peak within this window in the original segment
            search_segment = seg[start_idx:end_idx + 1]

            if len(search_segment) == 0:
                real_peak_indices[seg_idx] = None
                continue

            local_peak_idx = np.argmax(search_segment)
            actual_peak_idx = start_idx + local_peak_idx

            real_peak_indices[seg_idx] = actual_peak_idx
        else:
            real_peak_indices[seg_idx] = None

    for seg_idx, peak_idx in real_peak_indices.items():
        if peak_idx is not None and len(segments_all[seg_idx]) > 0:
            # Calculate real index in the original signal
            segment_start_t = qrs_indices[seg_idx] + int(
                0.12 * (rr_int[seg_idx]*fs))
            if seg_idx < len(rr_int):
                if rr_int[seg_idx] < 0.8:
                    segment_start_p = qrs_indices[seg_idx] + int(
                        # qrs_indices[seg_idx+1] - qrs_indices[seg_idx]=rr_int[seg_idx]
                        0.5 * (rr_int[seg_idx]*fs))
                elif 0.8 <= rr_int[seg_idx] < 1.2:
                    segment_start_p = qrs_indices[seg_idx] + int(
                        0.6 * (rr_int[seg_idx]*fs))-5
                elif 1.2 <= rr_int[seg_idx] < 1.78:
                    segment_start_p = qrs_indices[seg_idx] + int(
                        0.7 * rr_int[seg_idx]*fs)+30
                elif rr_int[seg_idx] >= 1.78:
                    segment_start_p = qrs_indices[seg_idx] + int(
                        0.7 * rr_int[seg_idx]*fs)+30
                else:
                    segment_start_p = qrs_indices[seg_idx] + int(
                        0.6 * rr_int[seg_idx]*fs)-5
            else:
                segment_start_p = qrs_indices[seg_idx] + int(
                    0.6 * rr_int[seg_idx]*fs)

            if dis:
                if det_t[seg_idx] is not None and dis_p[seg_idx]:  # type: ignore
                    real_idx = int(0.2*250) + peak_idx + \
                        det_t[seg_idx]  # type: ignore
                else:
                    real_idx = None
            else:
                if p:
                    real_idx = segment_start_p + peak_idx
                else:
                    real_idx = segment_start_t + peak_idx

            if real_idx is not None:
                if begin <= real_idx < begin + len(signal):
                    # plt.plot((real_idx-begin)/fs, signal[real_idx-begin], 'ro', markersize=6)
                    real_ind.append(real_idx)

    return real_ind


def predict_dis_p_waves_int(
    rr_intervals: List[float],
    pvc_labels: List[bool],
) -> List[bool]:
    """
    Predict segments with dissociated P-waves based on RR elongation.

    Parameters
    ----------
    rr_intervals : list of float
        RR intervals in seconds.
    pvc_labels : list of bool
        PVC flags per segment.

    Returns
    -------
    List[bool]
        True if RR suggests a dissociated P-wave.
    """
    # this detect for rr interval that can inlcude dissociated p peaks
    p_dis = []

    for i in range(1, len(rr_intervals)):
        if i == 1:
            p_dis.append(True)

        if p_dis[-1]:
            if (rr_intervals[i] > 0.8 * rr_intervals[i - 1]):
                check_criteria = True
            else:
                check_criteria = False
        else:
            check_criteria = (rr_intervals[i] > 1.6 * rr_intervals[i - 1] and
                              rr_intervals[i] > 1.6 and
                              not pvc_labels[i])
        p_dis.append(check_criteria)

    return p_dis


def p_dis_temp_gen(
    p_dis: List[bool],
    segments: List[np.ndarray],
    segments_dis: List[np.ndarray],
    qrs_indices: List[int],
    signal: np.ndarray,
    rr_int: List[float],
    det_t: List[int],
    det_p: List[int],
    begin: int,
    fs: int = 250,
    seg: bool = False,
    plot: bool = False
) -> List[int]:
    """
    Generate and localize dissociated P-waves from dissociated segments.

    Parameters
    ----------
    p_dis : list of bool
        Dissociated flags.
    segments : list of np.ndarray
        All original ECG segments.
    segments_dis : list of np.ndarray
        Candidate dissociated segments.
    qrs_indices : list of int
        R-peak locations.
    signal : np.ndarray
        ECG signal.
    rr_int : list of float
        RR intervals.
    det_t : list of int
        Detected T-wave peaks.
    det_p : list of int
        Detected P-wave peaks.
    begin : int
        Offset index in original signal.
    fs : int
        Sampling frequency.
    seg : bool
        If True, plot segments.
    plot : bool
        If True, plot templates.

    Returns
    -------
    List[int]
        Real indices in original signal of dissociated peaks.
    """
    temp_clus_comb_dis, num_comb_dis, scores_comb_dis, check_l_comb_dis, s_t_comb_dis = generate_template(
        segments_dis, 0.9)  # type: ignore
    updated_templates_dis, updated_counts_dis, updated_segment_map_dis, updated_segs_dis = drop_low_weight_templates(
        segments_dis, temp_clus_comb_dis, num_comb_dis, s_t_comb_dis, qrs_indices, 0.1)  # type: ignore
    peak_dis = detect_peaks_in_templates(
        updated_templates_dis, dis_p=True, plot=plot)
    real_dis = map_peaks_to_real_segments(signal, updated_templates_dis,
                                          segments, updated_segs_dis,  # type: ignore
                                          updated_segment_map_dis, peak_dis, False, True,
                                          qrs_indices, rr_int, begin, det_t=det_t, dis_p=p_dis)
    if seg:
        plot_templates_with_peaks(
            updated_templates_dis, peak_dis, window_start_ratio=0)
        plot_templates_with_segments(
            updated_segs_dis, updated_templates_dis, updated_segment_map_dis)
    return real_dis


####################
# plotting functions#
####################

def plot_templates_with_segments(segs, templates, segment_to_template):
    """
    Plots each ECG template and overlays the segments that were assigned to it.

    Helps visualize how well each template generalizes over its assigned segments.

    Parameters:
    - segs (list): Original signal segments.
    - templates (list): Final set of ECG templates.
    - segment_to_template (dict): Mapping of each segment to its template index.
    """
    num_templates = len(templates)
    # Define different colors for visualization
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(10, num_templates * 3))

    for temp_idx in range(num_templates):
        plt.subplot(num_templates, 1, temp_idx + 1)
        plt.title(f"Template {temp_idx} and Assigned Segments")

        plt.plot(templates[temp_idx], color='black', linewidth=3,
                 label=f'Template {temp_idx}', zorder=1000)

        for seg_idx, assigned_temp in segment_to_template.items():
            if assigned_temp == temp_idx:
                seg = segs[seg_idx]
                temp = templates[temp_idx]
                if len(seg) > len(temp):
                    seg_resampled = resample(seg, len(temp))
                else:
                    seg_resampled = seg
                plt.plot(seg_resampled, color=np.random.choice(
                    colors), alpha=0.6, linestyle='dashed')

        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_templates_with_peaks(templates, peak_indices, window_start_ratio=0.59):
    """
    Visualizes ECG templates along with detected peaks and optional detection windows.

    Parameters:
    - templates (list of np.array): ECG templates.
    - peak_indices (list of int or None): Indices of peaks detected in each template.
    - window_start_ratio (float): Ratio of the template where detection starts.
    """

    num_templates = len(templates)
    plt.figure(figsize=(10, 3 * num_templates))

    for i, (template, peak_idx) in enumerate(zip(templates, peak_indices)):
        plt.subplot(num_templates, 1, i + 1)
        plt.plot(template, label=f'Template {i}', linewidth=2)

        window_start = int(len(template) * window_start_ratio)

        if peak_idx is not None:
            plt.plot(peak_idx, template[peak_idx], 'ro',
                     label='Detected Peak', markersize=8)
            plt.axvline(peak_idx, color='r', linestyle='--', alpha=0.7)

        plt.title(f'Template {i} with Detection Window & Peak')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_real_peaks_in_ecg(
    signal, fs, begin, end, afib,
    normal_qrs=None, pvc_indices=None, q_beats=None,
    pause_indices=None, noisy_beats=None, afib_indices=None,
    real_peak_indices_t=None, detected_p=None
):
    """
    Plot ECG signal with annotated peaks and conditions.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal.
    fs : int
        Sampling frequency.
    begin : int
        Start index.
    end : int
        End index.
    afib : bool
        Whether to display AFIB zones.
    normal_qrs, pvc_indices, q_beats, pause_indices, noisy_beats : list of int
        Indexes of detected features.
    afib_indices, real_peak_indices_t, detected_p : list of int
        AFIB zone starts, T-peaks, and P-wave locations.
    """
    time = (np.arange(len(signal))+begin) / fs
    def to_time(arr): return [i / fs for i in arr]

    # function to filter by begin–end
    def filter_indices(indices):
        return [i for i in indices if begin <= i < end] if indices else []

    # Filter all index lists
    normal_qrs = filter_indices(normal_qrs)
    pvc_indices = filter_indices(pvc_indices)
    q_beats = filter_indices(q_beats)
    pause_indices = filter_indices(pause_indices)
    noisy_beats = filter_indices(noisy_beats)
    afib_indices = filter_indices(afib_indices)
    real_peak_indices_t = filter_indices(real_peak_indices_t)
    detected_p = filter_indices(detected_p)

    plt.figure(figsize=(15, 6))
    plt.plot(time, signal, label='ECG Signal', color='black', linewidth=1)

    if normal_qrs:
        plt.scatter(to_time(normal_qrs), signal[[
                    i - begin for i in normal_qrs]], color='green', label='Normal Beat', s=20)
    if pvc_indices:
        plt.scatter(to_time(pvc_indices), signal[[
                    i - begin for i in pvc_indices]], color='red', label='PVC', s=30)
    if q_beats:
        plt.scatter(to_time(q_beats), signal[[
                    i - begin for i in q_beats]], color='orange', label='Q Beat', s=30)
    if pause_indices:
        plt.scatter(to_time(pause_indices), signal[[
                    i - begin for i in pause_indices]], color='purple', label='Pause', marker='*', s=100)
    if noisy_beats:
        plt.scatter(to_time(noisy_beats), signal[[
                    i - begin for i in noisy_beats]], color='gray', label='Noisy Beat', marker='x', s=50)

    if afib_indices and afib:
        for i, t in enumerate(to_time(afib_indices)):
            start_time = max(t - 0.1, time[0])  # -begin/250
            end_time = min(t + 0.1, time[-1])  # -begin/250
            plt.axvspan(start_time, end_time, color='green',
                        alpha=0.7, label='AFIB Zone' if i == 0 else "")

    if real_peak_indices_t:
        plt.scatter(to_time(real_peak_indices_t), signal[[
                    i - begin for i in real_peak_indices_t]], color='blue', label='T Peaks', marker='x', s=50)

    if detected_p:
        plt.scatter(to_time(detected_p), signal[[
                    i - begin for i in detected_p]], color='green', label='P Peaks', marker='o', s=100)

    plt.title("ECG Signal with Annotations")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


########################
# acc checking functions#
########################

def safe_divide(numerator, denominator):
    if denominator == 0:
        return float('nan') if numerator == 0 else 0
    return numerator / denominator


def evaluate_detection(detected_p, annotated_p, tolerance, record_name, dataset):
    """
    Evaluates P-wave detection performance by comparing detected peaks with annotated ones.

    Parameters:
    ----------
    detected_p : list[int]
        Indices of detected P-wave peaks.
    annotated_p : list[int]
        Ground-truth annotated P-wave indices.
    tolerance : int
        Maximum allowed time difference (in samples) to consider a match.
    record_name : str
        Identifier for the ECG record (used in logging).

    Returns:
    -------
    dict
        Dictionary containing evaluation metrics and error statistics.
    list[int]
        Indices of false negatives (missed annotations).
    list[int]
        Indices of false positives (wrong detections).
    """
    matched_annot = set()
    matched_detected = set()
    matched_errors = []

    # --- Step 1: TP and matched sets ---
    for d in detected_p:
        found_match = False
        for a in annotated_p:
            if abs(d - a) <= tolerance:
                if a not in matched_annot:
                    matched_annot.add(a)
                    matched_detected.add(d)
                    matched_errors.append(d - a)
                    found_match = True
                    break

    TP = len(matched_detected)
    FN_indices = [a for a in annotated_p if all(
        abs(d - a) > tolerance for d in detected_p)]
    FP_indices = [d for d in detected_p if d not in matched_detected]

    FN = len(FN_indices)
    FP = len(FP_indices)

    accuracy = safe_divide(TP, TP + FP + FN)
    precision = safe_divide(TP, TP + FP)
    sensitivity = safe_divide(TP, TP + FN)
    f1 = 2 * safe_divide(precision * sensitivity, precision + sensitivity)

    print(dataset, f":{record_name}: Det_p: {len(detected_p)}, Ann_p: {len(annotated_p)}, TP,FP,FN: {TP},{FP},{FN}, F:{f1:.3f}, A:{accuracy:.3f}, P:{precision:.3f}, S:{sensitivity:.3f}")

    return {
        "record": record_name,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "n_detected": len(detected_p),
        "n_annotated": len(annotated_p)
    }, FN_indices, FP_indices


def algorithm(signal, indices, begin, fs=250, seg=False, p_co=0.8, plot=False, pt_v=0.1):
    """
    Main driver function for ECG P-wave detection using template generation and peak analysis.

    Parameters:
    - signal (np.array): Complete ECG signal.
    - indices tuple(tuple): qrs_indices, pvc_indices, afib_indices.
                            [qrs_indices, pvc_indices, afib_indices] per file.
    - begin (int): Start index for the segment to be analyzed.
    - fs (int): Sampling frequency.
    - seg (bool): Enable/disable debug plots.
    - p_co (float): Correlation threshold.
    - plot (bool): Plot results.
    - pt_v (float): Value for phasor transformation.

    Returns:
    - list: Detected P-wave indices.
    - list: Detected T-wave indices.
    """

    def to_time(arr): return [i / fs for i in arr]

    qrs_indices, pvc_indices, afib_indices = indices[0], indices[1], indices[2]

    # Median Filter with kernel size of 15
    filtered_ecg_butter = filter_ecg(signal, fs)
    rr_int = compute_rr_intervals(qrs_indices, fs)
    if seg:
        plt.figure(figsize=(15, 5))
        plt.plot(signal, label='raw')
        plt.plot(filtered_ecg_butter, label='filt')
        plt.title(f'raw and filt')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    pvc_labels = [i in set(pvc_indices) for i in qrs_indices]
    afib_labels = [i in set(afib_indices) for i in qrs_indices]

    pause_indices_new = [i for i in rr_int if i in qrs_indices and i >= 3]
    pause_labels = [i in set(pause_indices_new) for i in qrs_indices]

    valid_beats = pathology_check(
        qrs_indices, afib_labels, pvc_labels, pause_labels)
    segments, segments_p, segments_t, win_val = extract_segments(  # type: ignore
        filtered_ecg_butter, rr_int, qrs_indices, fs, valid_beats, begin)

    temp_clus_comb, num_comb, scores_comb, check_l_comb, s_t_comb = generate_template(
        segments, p_co)
    temp_clus_comb_p, num_comb_p, scores_comb_p, check_l_comb_p, s_t_comb_P = generate_template(
        segments_p, p_co)  # type: ignore
    temp_clus_comb_t, num_comb_t, scores_comb_t, check_l_comb_t, s_t_comb_T = generate_template(
        segments_t, p_co)

    updated_templates, updated_counts, updated_segment_map, updated_segs = drop_low_weight_templates(
        segments, temp_clus_comb, num_comb, s_t_comb, qrs_indices, 0.05)
    updated_templates_p, updated_counts_p, updated_segment_map_p, updated_segs_p = drop_low_weight_templates(
        segments_p, temp_clus_comb_p, num_comb_p, s_t_comb_P, qrs_indices, 0.05)  # type: ignore
    updated_templates_t, updated_counts_t, updated_segment_map_t, updated_segs_t = drop_low_weight_templates(
        segments_t, temp_clus_comb_t, num_comb_t, s_t_comb_T, qrs_indices, 0.05)

    peak_indices_p = detect_peaks_in_templates(
        updated_templates_p, pt_v=pt_v, plot=seg)

    if seg:
        plot_templates_with_peaks(
            updated_templates_p, peak_indices_p, window_start_ratio=0.59)
        plot_templates_with_segments(
            segments_p, updated_templates_p, updated_segment_map_p)
    peak_indices_t = detect_peaks_in_templates(
        updated_templates_t, pt_v=pt_v, plot=seg)
    real_peak_indices_p = map_peaks_to_real_segments(filtered_ecg_butter, updated_templates_p, segments,  # type: ignore
                                                     updated_segs_p, updated_segment_map_p, peak_indices_p, True, False, qrs_indices, rr_int, begin)  # type: ignore
    real_peak_indices_t = map_peaks_to_real_segments(filtered_ecg_butter, updated_templates_t, segments,  # type: ignore
                                                     updated_segs_t, updated_segment_map_t, peak_indices_t, False, False, qrs_indices, rr_int, begin)  # type: ignore
    real_peak_indices = np.sort(real_peak_indices_p + real_peak_indices_t)
    p_dis = predict_dis_p_waves_int(rr_int, pvc_labels,)  # type: ignore
    segments_dis, t_val = extract_segments(  # type: ignore
        filtered_ecg_butter, rr_int, qrs_indices, fs, p_dis, begin, True, real_peak_indices_t, real_peak_indices_p)  # type: ignore
    dis_p = p_dis_temp_gen(p_dis, segments, segments_dis, qrs_indices,  # type: ignore
                           filtered_ecg_butter, rr_int, t_val, real_peak_indices_p, begin, seg=seg, plot=seg)  # type: ignore
    all_p_indices = real_peak_indices_p + dis_p
    all_p_indices.sort()
    all_p_times = [i/fs for i in all_p_indices]

    detected_p = np.array(all_p_indices).tolist()

    window_size = 10

    def shifted(signal):
        if np.any(signal < 0):
            return signal - np.min(signal)
        else:
            return np.copy(signal)

    rr_p = rr_for_p(detected_p, qrs_indices)
    rr_t = rr_for_p(real_peak_indices_t, qrs_indices)
    filtered_p_after_t = []

    for i in range(len(rr_int)):
        # Get all P-peaks and T-peaks in this RR interval
        p_in_rr = [detected_p[j] for j in range(len(rr_p)) if rr_p[j] == i]
        t_in_rr = [real_peak_indices_t[j]
                   for j in range(len(rr_t)) if rr_t[j] == i]

        if not t_in_rr:
            t_in_rr = [None]  # Skip if no T-peak in this RR interval

        t_peak = t_in_rr[0]  # Assume one T per RR (or take first if multiple)

        # Keep only P-peaks that occur after the T-peak
        for p_peak in p_in_rr:
            if t_peak is not None and p_peak > t_peak:
                filtered_p_after_t.append(p_peak)
            elif t_peak is None:
                filtered_p_after_t.append(p_peak)

    detected_p = filtered_p_after_t

    filtered_peaks = []

    for i in range(len(detected_p)):
        p_idx = detected_p[i]-begin
        left_idx = max(0, p_idx - window_size)
        right_idx = min(len(filtered_ecg_butter), (p_idx + window_size + 1))

        l_seg = filtered_ecg_butter[left_idx:p_idx+1]
        r_seg = filtered_ecg_butter[p_idx:right_idx]
        l_seg = l_seg-np.min(l_seg)
        r_seg = r_seg-np.min(r_seg)

        l_amp = (np.max(shifted(l_seg)) if len(l_seg) > 0 else 0)
        r_amp = (np.max(shifted(r_seg)) if len(r_seg) > 0 else 0)
        amp = max(l_amp, r_amp)

        nearest_r = min(qrs_indices, key=lambda r: abs(r - (p_idx+begin)))
        r_amp_val = abs(filtered_ecg_butter[nearest_r-begin])

        # Keep only if amp ≥ percentage of nearest R-peak amplitude
        if r_amp_val <= 0.15:
            if rr_int[rr_p[i]] >= 0.9:
                if amp >= 0.09 * r_amp_val:
                    filtered_peaks.append(detected_p[i])
            else:
                if amp >= 0.3 * r_amp_val:
                    filtered_peaks.append(detected_p[i])
        elif 0.15 <= r_amp_val < 0.2:
            if rr_int[rr_p[i]] >= 0.9:
                if amp >= 0.07 * r_amp_val:
                    filtered_peaks.append(detected_p[i])
            else:
                if amp >= 0.2 * r_amp_val:
                    filtered_peaks.append(detected_p[i])
        elif 0.2 <= r_amp_val < 0.3:
            if rr_int[rr_p[i]] >= 0.9:
                if amp >= 0.07 * r_amp_val:
                    filtered_peaks.append(detected_p[i])
            else:
                if amp >= 0.1 * r_amp_val:
                    filtered_peaks.append(detected_p[i])
        elif 0.3 < r_amp_val < 0.45:
            if amp >= 0.03 * r_amp_val:
                filtered_peaks.append(detected_p[i])
        elif 0.45 <= r_amp_val <= 0.7:
            if amp >= 0.025 * r_amp_val:
                filtered_peaks.append(detected_p[i])
        elif 1.1 <= r_amp_val <= 1.8:
            if amp >= 0.02 * r_amp_val:
                filtered_peaks.append(detected_p[i])
        elif 1.8 < r_amp_val:
            if amp >= 0.01 * r_amp_val:
                filtered_peaks.append(detected_p[i])
        else:
            if amp >= 0.03 * r_amp_val:
                filtered_peaks.append(detected_p[i])

    detected_p = filtered_peaks  # overwrite original

    return detected_p, real_peak_indices_t


def map_r_peaks_from_both_sources(signal_dict, indices_dict, sample_r, record_name=None, tolerance=15):
    """
    Matches annotated R-peaks from load_ann with sample_r_dict and extracts valid signal segments
    where matched peaks have both previous and next matches.

    Parameters:
    -----------
    signal_dict : dict
        Dictionary of ECG signals from load_ann.
    indices_dict : dict
        Dictionary of indices from load_ann (qrs_indices).
    sample_r : dict
        Dictionary of reference R-peaks (e.g., from WFDB annotations).
    record_name : str or None
        Optional specific record to process.
    tolerance : int
        Maximum sample distance for matching R-peaks.

    Returns:
    --------
    segments_dict : dict
        Dictionary containing valid signal segments keyed by record name.
    segment_boundaries : dict
        Dictionary of (start, end) tuples per segment keyed by record name.
    """
    segments_dict = {}
    segment_boundaries = {}

    target_records = [record_name] if record_name else signal_dict.keys()

    for rec in target_records:
        rec_filename = rec if rec.endswith('.h5py') else rec + '.h5py'
        signal = signal_dict.get(rec_filename)
        ann_r_peaks = indices_dict.get(rec_filename, [[], [], []])[
            0]  # qrs_indices

        if signal is None or len(ann_r_peaks) == 0 or len(sample_r) == 0:
            # print('no')
            continue

        matched_r = []
        for r in ann_r_peaks:
            diffs = np.abs(np.array(sample_r) - r)
            closest_idx = np.argmin(diffs)
            if diffs[closest_idx] <= tolerance:
                matched_r.append(r)

        matched_r = sorted(set(matched_r))
        valid_segments = []
        boundaries = []

        for i in range(len(matched_r)):
            r_curr = matched_r[i]
            if r_curr in ann_r_peaks:
                k = ann_r_peaks.index(r_curr)
            else:
                k = -1  # or handle as needed

            if k > 0:
                r_prev = ann_r_peaks[k - 1]
            else:
                r_prev = r_curr
            boundaries.append((r_prev, r_curr))

        # segments_dict[rec] = valid_segments
        segment_boundaries[rec] = boundaries

    return segments_dict, segment_boundaries


def filter_peaks_within_segments(peaks, segments):
    """
    Filters the given peak indices to retain only those that lie within any of the provided segments.

    Parameters:
    -----------
    peaks : list of int
        Detected or annotated peak indices.
    segments : list of tuple
        List of (start, end) index tuples defining valid segments.

    Returns:
    --------
    filtered_peaks : list of int
        Peaks that fall within any of the segments.
    """
    filtered_peaks = []
    for peak in peaks:
        for start, end in segments:
            # print(start,peak,end)
            if start <= peak < end:
                filtered_peaks.append(peak)
                break
    return filtered_peaks


def load_ann(folder_path=None, record_name=None, begin=0, end=None):
    """
    Loads ECG signals and their annotations from HDF5 files.

    Parameters:
    ----------
    folder_path : str
        Path to the folder containing ECG HDF5 files.
    record_name : str or list of str
        Specific file(s) to load. If None, all .h5/.h5py files in folder are used.
    begin : int
        Starting sample index to extract.
    end : int
        Ending sample index to extract (exclusive). If None, goes till end.

    Returns:
    -------
    annotation : dict
        Raw annotation arrays.
    signal : dict
        Extracted ECG signals.
    indices : dict
        [qrs_indices, pvc_indices, afib_indices] per file.
    records : list
        Processed filenames.
    """
    annotation = {}
    signal = {}
    indices = {}
    records = []
    if folder_path is not None:
        # Load from a single file if record_name is specified
        if record_name is not None:
            file_list = record_name
        else:
            file_list = sorted(os.listdir(folder_path))
        records = file_list
        for file_name in file_list:
            if file_name.lower().endswith(('.h5', '.h5py')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with h5py.File(file_path, 'r') as f:
                        esignal = np.array(f['signal'])
                        ann = np.array(f['annotation'])
                        if end is not None:
                            esignal = esignal[begin:end]
                        else:
                            esignal = esignal[begin:]
                            end = len(esignal)
                        annotation[file_name] = ann
                        signal[file_name] = esignal
                        subset_ann = [
                            row for row in annotation[file_name] if begin <= row[0] < end]

                        # Extract beat types
                        qrs_indices = [
                            int(row[0]) for row in subset_ann if row[1] in [78, 86, 81]]
                        normal_qrs = [int(row[0])
                                      for row in subset_ann if row[1] == 78]
                        pvc_indices = [int(row[0])
                                       for row in subset_ann if row[1] == 86]
                        q_beats = [int(row[0])
                                   for row in subset_ann if row[1] == 81]
                        afib_indices = [int(row[0])
                                        for row in subset_ann if row[2] == 7]

                        indices[file_name] = [
                            qrs_indices, pvc_indices,  afib_indices]

                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    annotation[file_name] = []
                    signal[file_name] = []
                    indices[file_name] = []

    else:
        print("Please provide a folder_path.")
        return {}, {}, {}

    return annotation, signal, indices, records


def acc_check_wfdb_ann(dataset, det_p, record_name=None, begin=0, end=None, tolerance=19, signal=None, indices=None):
    """
    Compares detected P-wave peaks with WFDB annotations and computes metrics.

    Parameters:
    ----------
    dataset : str
        One of ['mitbih', 'ludb', 'butdb', 'qtdb'].
    det_p : dict
        Detected P peaks, per record.
    record_name : str, optional
        Single record to process. If None, processes entire dataset.
    begin : int
        Start sample index.
    end : int
        End sample index (exclusive).
    tolerance : int
        Allowed tolerance (in samples) for matching peaks.
    signal : dict, optional
    indices : dict, optional

    Returns:
    -------
    acc : list of dicts
        Accuracy metrics for each record.
    last_fp : int
        Last record's FP count.
    last_fn : int
        Last record's FN count.
    annotation : dict
        Reference annotations per record.
    fn : list
        List of FN peak indices.
    fp : list
        List of FP peak indices.
    """
    # --- Sampling Config ---
    fs_ann = 360
    fs_sig = 250
    resample_ratio = fs_sig / fs_ann

    # --- Dataset Setup ---
    if dataset == 'mitbih':
        annotation_file = 'pwave'
        pn_dir = 'pwave'
    elif dataset == 'ludb':
        annotation_file = 'ii'
        pn_dir = 'ludb/data'
    elif dataset == 'butdb':
        annotation_file = 'pwave'
        pn_dir = 'but-pdb'
    elif dataset == 'qtdb':
        annotation_file = 'q1c'
        pn_dir = 'qtdb'
    else:
        print("Invalid dataset name.")
        return [], [], [], {}

    # --- Record list ---
    if record_name:
        record_clean = os.path.splitext(record_name)[0]
        record_list = [f'{record_clean}']
    else:
        record_list = wfdb.get_record_list(pn_dir)

    # --- Step 1: Load annotations ---
    annotation = {}
    p_symbols = ['p', 'n']
    for rec in record_list:
        try:
            ann = wfdb.rdann(rec, annotation_file, pn_dir=pn_dir)
            # print(ann.symbol)
            # Special resampling for BUTDB low-rate files
            if rec in ['39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50'] and dataset == 'butdb':
                resample_ratio = 250 / 128
            elif dataset == 'ludb':
                resample_ratio = 250 / 500
                p_symbols = ['p']

            elif dataset == 'qtdb':
                resample_ratio = 250 / 250
                p_symbols = ['p']

            samples = [
                int(s * resample_ratio)
                for s, sym in zip(ann.sample, ann.symbol)  # type: ignore
                if sym.lower() in p_symbols and
                (res := int(s * resample_ratio)) >= begin and
                (end is None or res < end)
            ]
            annotation[rec] = samples
        except Exception as e:
            print(f"Error reading annotations for {rec}: {e}")
            annotation[rec] = []

    # --- Step 2: Evaluate ---
    acc = []
    last_fp = []
    last_fn = []

    for rec_key, ann_samples in annotation.items():
        if rec_key not in det_p:
            print(f"Warning: No detection found for {rec_key}")
            continue

        detected = [d for d in det_p[rec_key] if d >=
                    begin and (end is None or d < end)]
        if dataset == 'ludb' or dataset == 'qtdb':
            samples_r = [
                int(s * resample_ratio)
                for s, sym in zip(ann.sample, ann.symbol)  # type: ignore
                if sym.lower() in ['n'] and
                (res := int(s * resample_ratio)) >= begin and
                (end is None or res < end)
            ]
            segm, segm_boundaries = map_r_peaks_from_both_sources(
                signal, indices, samples_r, record_name=record_name)

            if dataset == 'qtdb' or dataset == 'ludb':
                if samples_r:
                    segments = segm_boundaries.get(rec_key, [])
                    detected = filter_peaks_within_segments(
                        det_p[rec_key], segments)
                    ann_samples = filter_peaks_within_segments(
                        ann_samples, segments)
                    if dataset == 'ludb':
                        detected = [d for d in detected if d >=
                                    segments[0][1] and d <= segments[-1][1]]
                        ann_samples = [d for d in ann_samples if d >=
                                       segments[0][1] and d <= segments[-1][1]]
            else:
                if samples_r:
                    detected = [d for d in det_p[rec_key] if d >=
                                samples_r[0] and d <= samples_r[-1]]
                    ann_samples = [d for d in ann_samples if d >=
                                   samples_r[0] and d <= samples_r[-1]]
        elif dataset == 'butdb':
            ann1 = wfdb.rdann(rec_key, 'qrs', pn_dir=pn_dir)
            samples_r1 = [
                int(s * resample_ratio)
                for s, sym in zip(ann1.sample, ann1.symbol)  # type: ignore
                if sym.lower() in ['n'] and
                (res := int(s * resample_ratio)) >= begin and
                (end is None or res < end)]
            if samples_r1:
                detected = [d for d in det_p[rec_key] if d >=
                            samples_r1[0] and d <= samples_r1[-1]]
                ann_samples = [d for d in ann_samples if d >=
                               samples_r1[0] and d <= samples_r1[-1]]

        res_dict, fn, fp = evaluate_detection(
            detected, ann_samples, tolerance, rec_key, dataset=dataset)

        acc.append(res_dict)
        last_fp = res_dict["FP"]
        last_fn = res_dict["FN"]

    return acc, last_fp, last_fn, annotation, fn, fp  # type: ignore


def detect_p(
    dataset=None,
    folder_path=None,
    record_name=None,
    begin=0,
    end=None,
    plot=False,
    inspect=False,
    fs=250,
    seg=False,
    p_co=0.9,
    pt_v=0.1
):
    """
    Detects P-wave peaks in ECG signals using a custom detection algorithm and optionally evaluates performance.

    Steps:
    1. Loads ECG signals and annotations using `load_ann()`.
    2. Runs the custom detection algorithm on each ECG signal.
    3. Optionally visualizes the results and evaluates detection performance using reference annotations.

    Parameters:
    ----------
    dataset : str, optional
        Name of the dataset (e.g., 'mitbih', 'butdb', 'ludb') for WFDB annotation-based evaluation.
    folder_path : str, optional
        Path to the folder containing .h5/.h5py ECG files.
    record_name : str or list of str, optional
        Name(s) of specific ECG records to process. If None, all valid files in the folder are used.
    begin : int, optional
        Start time (in seconds) for ECG processing.
    end : int, optional
        End time (in seconds). If None, processes until the end of the signal.
    plot : bool, optional
        If True, plots the ECG with detected P/T waves and reference peaks.
    inspect : bool, optional
        If True, performs visual inspection using `visual_inspection()`.
    fs : int, optional
        Sampling frequency of the ECG (default is 250 Hz).
    seg : bool, optional
        Enables segmented detection mode if True.
    p_co : float, optional
        Pearson correlation threshold for template matching.
    pt_v : float, optional
        Reference value used in phasor transformation.

    Returns:
    -------
    pdet : dict
        Dictionary mapping record name to list of detected P-wave indices.
    data : dict
        Dictionary mapping record name to evaluation metrics (only if `dataset` is provided).

    Notes:
    -----
    - This function multiplies signal values by 10,00 only for mit bih due to decreased values prevviously before passing them to the algorithm.
    - Accuracy metrics are computed only if a dataset name is provided.
    - Time indices (`begin`, `end`) are converted from seconds to sample points using the sampling frequency.
    - Assumes the availability of the functions: `load_ann`, `algorithm`, `plot_real_peaks_in_ecg`, `visual_inspection`, and `acc_check_wfdb_ann`.
    """
    data = {}
    pdet = {}
    begin = begin*250
    if end is None:
        end = end
    else:
        end = end*250
    annotation, signal, indices, records = load_ann(  # type: ignore
        folder_path=folder_path, record_name=record_name, begin=begin, end=end)

    for rec in records:

        if end is None:
            end = len(signal[rec].ravel())
        k = 1000 if dataset == 'mitbih' else 1
        det_p, det_t = algorithm(signal[rec].ravel(
        )*k, indices[rec], begin, fs, seg, p_co, plot=plot, pt_v=pt_v)
        if inspect:
            visual_inspection(signal[rec].ravel(),  # type: ignore
                              annotation[rec], rec, 30, fs)
        if dataset is not None:
            acc, fpl, fnl, ann, fn, fp = acc_check_wfdb_ann(dataset, {os.path.splitext(
                rec)[0]: det_p}, record_name=os.path.splitext(rec)[0], begin=begin, end=end, tolerance=19, signal=signal, indices=indices)
            data[rec] = acc

        if plot:
            plot_real_peaks_in_ecg(signal[rec].ravel()*k, fs, begin, end, True, normal_qrs=indices[rec][0], afib_indices=indices[rec][2],
                                   real_peak_indices_t=det_t, detected_p=det_p, pvc_indices=indices[rec][1])

        if plot and dataset is not None:
            plot_real_peaks_in_ecg(signal[rec].ravel(), fs, begin, end, True, normal_qrs=indices[rec][0], afib_indices=indices[rec][2],
                                   real_peak_indices_t=fn, detected_p=fp,  # type: ignore
                                   pvc_indices=indices[rec][1])

        pdet[rec] = det_p

    return pdet, data


def xcel(data, dataset, exclude_records=[]):
    """
    Generate summary statistics, filter poor-performing records, and export results to Excel.

    Parameters:
    ----------
    data : dict
        Dictionary of evaluation results per record. Each value should be a list with one result dict.
    dataset : str
        Name of the dataset (used for file naming and filtering thresholds).
    exclude_records : list, optional
        List of record names to exclude from processing.

    Returns:
    -------
    None
    """
    # Format: data = {record_name: [result_dict]}
    data = {k: v for k, v in data.items() if k not in exclude_records}
    flat_data = [v[0] for v in data.values()]
    df = pd.DataFrame(flat_data)

    # convert columns ---
    numeric_cols = [
        'accuracy', 'precision', 'sensitivity', 'f1',
        'TP', 'FP', 'FN', 'n_detected', 'n_annotated'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # --- Filter low-performing records ---
    if dataset == 'qtdb':
        filtered_df = df[

            (df['sensitivity'] <= 0.9) |
            (df['precision'] <= 0.8)]
    else:
        filtered_df = df[

            (df['sensitivity'] <= 0.9) |
            (df['precision'] <= 0.9)]
    filtered_record_names = [
        name + ".h5py" for name in filtered_df['record'].tolist()]

    print("Filtered Record Names:", filtered_record_names)

    # --- Compute average metrics ---
    avg_metrics = df[['accuracy', 'precision', 'sensitivity', 'f1']].mean(
        skipna=True).round(3)

    # --- Gross metrics ---
    df_no_nan = df.dropna(subset=['TP', 'FP', 'FN'])
    df_no_nan[['TP', 'FP', 'FN']] = df_no_nan[['TP', 'FP', 'FN']].astype(int)

    total_TP = df_no_nan['TP'].sum()
    total_FP = df_no_nan['FP'].sum()
    total_FN = df_no_nan['FN'].sum()

    gross_accuracy = round(total_TP / (total_TP + total_FP + total_FN),
                           3) if (total_TP + total_FP + total_FN) else 0
    gross_ppv = round(total_TP / (total_TP + total_FP),
                      3) if (total_TP + total_FP) else 0
    gross_sensitivity = round(
        total_TP / (total_TP + total_FN), 3) if (total_TP + total_FN) else 0

    # --- Export to Excel ---
    excel_filename = "p_wave_detection_results" + dataset + ".xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="All_Data", index=False)
        worksheet = writer.sheets["All_Data"]
        row_count = len(df) + 1

        # Average row
        worksheet.write(row_count, 0, "Average")
        worksheet.write(row_count, 1, avg_metrics['accuracy'])
        worksheet.write(row_count, 2, avg_metrics['precision'])
        worksheet.write(row_count, 3, avg_metrics['sensitivity'])
        worksheet.write(row_count, 4, avg_metrics['f1'])

        # Gross row
        worksheet.write(row_count + 1, 0, "Gross")
        worksheet.write(row_count + 1, 1, gross_accuracy)
        worksheet.write(row_count + 1, 2, gross_ppv)
        worksheet.write(row_count + 1, 3, gross_sensitivity)

        # Filtered records
        filtered_df.to_excel(writer, sheet_name="Filtered_Data", index=False)

    print(f"\n Results saved to: {excel_filename}")
    print(
        f" Averages — Acc: {avg_metrics['accuracy']:.4f}, PPV: {avg_metrics['precision']:.4f}, Sens: {avg_metrics['sensitivity']:.4f}, F1: {avg_metrics['f1']:.4f}")
    print(
        f" Gross   — Acc: {gross_accuracy:.4f}, PPV: {gross_ppv:.4f}, Sens: {gross_sensitivity:.4f}")


def bih(folder_path, record_name=None, begin=0, end=None, plot=False, seg=False, p_co=0.7, pt_v=0.1):
    if record_name is None:
        g = ['100.h5py', '101.h5py', '103.h5py', '106.h5py', '117.h5py', '119.h5py',
             '122.h5py', '207.h5py', '214.h5py', '222.h5py', '223.h5py', '231.h5py']

    else:
        g = record_name
    det_p, data = detect_p(dataset='mitbih', folder_path=folder_path, record_name=g,
                           begin=begin, end=end, plot=plot, inspect=False, fs=250, seg=seg, p_co=p_co, pt_v=pt_v)
    if record_name is None:
        print('for phisiological signals')
        xcel(data, dataset='mitbih', exclude_records=[
            '106.h5py', '119.h5py', '207.h5py', '214.h5py', '222.h5py', '223.h5py', '231.h5py'])
        print('for non-physiological signals')
        xcel(data, dataset='mitbih', exclude_records=[
            '100.h5py', '101.h5py', '103.h5py', '117.h5py', '122.h5py'])
        print('without 207 signal in MIT BIH PWAVE DB')
        xcel(data, dataset='mitbih', exclude_records=['207.h5py'])
        print('for all 12 signals in MIT BIH PWAVE DB')
        xcel(data, dataset='mitbih')

    return None


def but(folder_path, record_name=None, begin=0, end=None, plot=False, seg=False, p_co=0.7, pt_v=0.1):
    if record_name is None:
        g = ['01.h5py', '02.h5py', '03.h5py', '04.h5py', '05.h5py', '06.h5py', '07.h5py', '08.h5py', '09.h5py', '10.h5py',
             '11.h5py', '12.h5py', '13.h5py', '14.h5py', '15.h5py', '16.h5py', '17.h5py', '18.h5py', '19.h5py', '20.h5py',
             '21.h5py', '22.h5py', '23.h5py', '24.h5py', '25.h5py', '26.h5py', '27.h5py', '28.h5py', '29.h5py', '30.h5py',
             '31.h5py', '32.h5py', '33.h5py', '34.h5py', '35.h5py', '36.h5py', '37.h5py', '38.h5py', '39.h5py', '40.h5py',
             '41.h5py', '42.h5py', '43.h5py', '44.h5py', '45.h5py', '46.h5py', '47.h5py', '48.h5py', '49.h5py', '50.h5py']
    else:
        g = record_name
    det_p, data = detect_p(dataset='butdb', folder_path=folder_path, record_name=g,
                           begin=begin, end=end, plot=plot, inspect=False, fs=250, seg=seg, p_co=p_co, pt_v=pt_v)
    if record_name is None:
        xcel(data, dataset='butdb')

    return None


def ludb(folder_path, record_name=None, begin=0, end=None, plot=False, seg=False, p_co=0.7, pt_v=0.1):
    if record_name is None:
        g = ['1.h5py', '2.h5py', '3.h5py', '4.h5py', '5.h5py', '6.h5py', '7.h5py', '8.h5py', '9.h5py', '10.h5py',
             '11.h5py', '12.h5py', '13.h5py', '14.h5py', '15.h5py', '16.h5py', '17.h5py', '18.h5py', '19.h5py', '20.h5py',
             '21.h5py', '22.h5py', '23.h5py', '24.h5py', '25.h5py', '26.h5py', '27.h5py', '28.h5py', '29.h5py', '30.h5py',
             '31.h5py', '32.h5py', '33.h5py', '34.h5py', '35.h5py', '36.h5py', '37.h5py', '38.h5py', '39.h5py', '40.h5py',
             '41.h5py', '42.h5py', '43.h5py', '44.h5py', '45.h5py', '46.h5py', '47.h5py', '48.h5py', '49.h5py', '50.h5py',
             '51.h5py', '52.h5py', '53.h5py', '54.h5py', '55.h5py', '56.h5py', '57.h5py', '58.h5py', '59.h5py', '60.h5py',
             '61.h5py', '62.h5py', '63.h5py', '64.h5py', '65.h5py', '66.h5py', '67.h5py', '68.h5py', '69.h5py', '70.h5py',
             '71.h5py', '72.h5py', '73.h5py', '74.h5py', '75.h5py', '76.h5py', '77.h5py', '78.h5py', '79.h5py', '80.h5py',
             '81.h5py', '82.h5py', '83.h5py', '84.h5py', '85.h5py', '86.h5py', '87.h5py', '88.h5py', '89.h5py', '90.h5py',
             '91.h5py', '92.h5py', '93.h5py', '94.h5py', '95.h5py', '96.h5py', '97.h5py', '98.h5py', '99.h5py', '100.h5py',
             '101.h5py', '102.h5py', '103.h5py', '104.h5py', '105.h5py', '106.h5py', '107.h5py', '108.h5py', '109.h5py', '110.h5py',
             '111.h5py', '112.h5py', '113.h5py', '114.h5py', '115.h5py', '116.h5py', '117.h5py', '118.h5py', '119.h5py', '120.h5py',
             '121.h5py', '122.h5py', '123.h5py', '124.h5py', '125.h5py', '126.h5py', '127.h5py', '128.h5py', '129.h5py', '130.h5py',
             '131.h5py', '132.h5py', '133.h5py', '134.h5py', '135.h5py', '136.h5py', '137.h5py', '138.h5py', '139.h5py', '140.h5py',
             '141.h5py', '142.h5py', '143.h5py', '144.h5py', '145.h5py', '146.h5py', '147.h5py', '148.h5py', '149.h5py', '150.h5py',
             '151.h5py', '152.h5py', '153.h5py', '154.h5py', '155.h5py', '156.h5py', '157.h5py', '158.h5py', '159.h5py', '160.h5py',
             '161.h5py', '162.h5py', '163.h5py', '164.h5py', '165.h5py', '166.h5py', '167.h5py', '168.h5py', '169.h5py', '170.h5py',
             '171.h5py', '172.h5py', '173.h5py', '174.h5py', '175.h5py', '176.h5py', '177.h5py', '178.h5py', '179.h5py', '180.h5py',
             '181.h5py', '182.h5py', '183.h5py', '184.h5py', '185.h5py', '186.h5py', '187.h5py', '188.h5py', '189.h5py', '190.h5py',
             '191.h5py', '192.h5py', '193.h5py', '194.h5py', '195.h5py', '196.h5py', '197.h5py', '198.h5py', '199.h5py', '200.h5py']
    else:
        g = record_name
    det_p, data = detect_p(dataset='ludb', folder_path=folder_path, record_name=g, begin=begin,
                           end=end, plot=plot, inspect=False, fs=250, seg=seg, p_co=p_co, pt_v=pt_v)
    if record_name is None:
        xcel(data, dataset='ludb')

    return None


def qtdb(folder_path, record_name=None, begin=0, end=None, plot=False, seg=False, p_co=0.7, pt_v=0.1):
    if record_name is None:
        g = ['sel100.h5py',  'sel103.h5py',  'sel114.h5py', 'sel116.h5py',
             'sel117.h5py', 'sel123.h5py', 'sel14046.h5py', 'sel14157.h5py', 'sel14172.h5py', 'sel15814.h5py',
             'sel16265.h5py', 'sel16272.h5py', 'sel16273.h5py', 'sel16420.h5py', 'sel16483.h5py', 'sel16539.h5py',
             'sel16773.h5py', 'sel16786.h5py', 'sel16795.h5py', 'sel17152.h5py', 'sel17453.h5py', 'sel213.h5py',
             'sel223.h5py', 'sel230.h5py', 'sel231.h5py',  'sel233.h5py', 'sel30.h5py',
             'sel301.h5py', 'sel302.h5py', 'sel306.h5py', 'sel307.h5py', 'sel308.h5py', 'sel31.h5py',
             'sel32.h5py', 'sel33.h5py', 'sel34.h5py', 'sel35.h5py',  'sel38.h5py',
             'sel39.h5py', 'sel40.h5py', 'sel41.h5py', 'sel42.h5py', 'sel43.h5py', 'sel44.h5py', 'sel45.h5py',
             'sel46.h5py', 'sel47.h5py', 'sel48.h5py', 'sel49.h5py',  'sel51.h5py', 'sel52.h5py',
             'sel803.h5py', 'sel808.h5py', 'sel811.h5py', 'sel820.h5py', 'sel821.h5py', 'sel840.h5py', 'sel847.h5py',
             'sel853.h5py', 'sel871.h5py', 'sel872.h5py', 'sel873.h5py', 'sel883.h5py', 'sel891.h5py', 'sele0104.h5py',
             'sele0106.h5py', 'sele0107.h5py',  'sele0111.h5py', 'sele0112.h5py',
             'sele0116.h5py', 'sele0121.h5py', 'sele0122.h5py', 'sele0124.h5py', 'sele0126.h5py', 'sele0129.h5py',
             'sele0133.h5py', 'sele0136.h5py', 'sele0166.h5py', 'sele0170.h5py', 'sele0203.h5py', 'sele0210.h5py',
             'sele0211.h5py', 'sele0303.h5py', 'sele0405.h5py', 'sele0406.h5py', 'sele0409.h5py', 'sele0411.h5py',
             'sele0509.h5py', 'sele0603.h5py', 'sele0604.h5py', 'sele0606.h5py', 'sele0607.h5py', 'sele0609.h5py',
             'sele0612.h5py', 'sele0704.h5py']
    else:
        g = record_name
    det_p, data = detect_p(dataset='qtdb', folder_path=folder_path, record_name=g, begin=begin,
                           end=end, plot=plot, inspect=False, fs=250, seg=seg, p_co=p_co, pt_v=pt_v)
    if record_name is None:
        xcel(data, dataset='qtdb')

    return None


p = 0.8
