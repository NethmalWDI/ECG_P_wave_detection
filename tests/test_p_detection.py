import pandas as pd
from unittest.mock import patch, MagicMock

import unittest
import numpy as np



class TestECG(unittest.TestCase):

    def setUp(self): # type: ignore
        self.fs = 250  # Sampling frequency in Hz

    def test_filter_ecg_valid_output(self):
        """
        Test that filter_ecg returns an output of the same shape and type 
        as the input signal, ensuring that filtering does not alter signal dimensions.
        """
        # Create a synthetic ECG-like signal: sine wave + noise
        from p_detection import filter_ecg
        duration_sec = 5
        t = np.linspace(0, duration_sec, duration_sec * self.fs)
        signal = np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(len(t))

        # Apply the filter
        filtered = filter_ecg(signal, self.fs)

        # Assertions
        self.assertIsInstance(filtered, np.ndarray)
        self.assertEqual(filtered.shape, signal.shape)
        self.assertFalse(np.isnan(filtered).any(),
                         "Filtered signal contains NaNs.")

    def test_compute_rr_intervals(self):
        """
        Test that RR intervals are computed correctly from sample R-peak indices.
        """
        from p_detection import compute_rr_intervals

        fs = 250  # Hz
        r_peaks = [100, 350, 600, 850]  # Each 1s apart → 1.0s RR intervals
        expected_rr = np.array([1.0, 1.0, 1.0])

        rr = compute_rr_intervals(r_peaks, fs)

        np.testing.assert_array_almost_equal(rr, expected_rr, decimal=3)
        self.assertEqual(len(rr), len(r_peaks) - 1)

    def test_phasor_transform_output_shape(self):
        """
        Test that phasor_transform returns an array of the same shape as input.
        """
        from p_detection import phasor_transform

        ecg = np.array([0.0, 0.5, 1.0, 1.5])
        rv = 1.0
        transformed = phasor_transform(ecg, rv)

        self.assertEqual(transformed.shape, ecg.shape)
        np.testing.assert_allclose(transformed, np.arctan(ecg / rv), atol=1e-6)

    def test_phasor_transform_invalid_input(self):
        """
        Test phasor_transform raises errors for invalid inputs.
        """
        from p_detection import phasor_transform

        with self.assertRaises(ValueError):
            phasor_transform(np.array([[1, 2], [3, 4]]), rv=1.0)  # Not 1D

        with self.assertRaises(ValueError):
            phasor_transform(np.array([1, 2, 3]), rv=0.0)  # Zero reference


    def test_pathology_check_filters_pathological(self):
        """
        Test that pathology_check filters out AFIB, PVC, and pause-labeled R-peaks.
        """
        from p_detection import pathology_check

        r_peaks = [100, 200, 300, 400]
        afib_labels = [False, True, False, False]
        pvc_labels = [False, False, True, False]
        pause_labels = [False, False, False, True]

        expected = [100, None, None, None]
        result = pathology_check(r_peaks, afib_labels, pvc_labels, pause_labels)

        self.assertEqual(result, expected)


    def test_pathology_check_all_valid(self):
        """Test pathology_check when all beats are normal."""
        from p_detection import pathology_check

        r_peaks = [100, 200, 300]
        flags = [False, False, False]
        result = pathology_check(r_peaks, flags, flags, flags)

        self.assertEqual(result, r_peaks)


    def test_pathology_check_input_mismatch(self):
        """Test pathology_check raises error for mismatched input lengths."""
        from p_detection import pathology_check

        with self.assertRaises(ValueError):
            pathology_check([100, 200], [True], [False, False], [False, False])


    def test_rr_for_p_correct_assignment(self):
        """
        Test that each P-peak is correctly assigned to the RR interval it falls within.
        """
        from p_detection import rr_for_p

        r_peaks = [100, 200, 300, 400]
        p_peaks = [150, 250, 350]

        expected = [0, 1, 2]
        result = rr_for_p(p_peaks, r_peaks) 

        self.assertEqual(result, expected)


    def test_rr_for_p_outside_intervals(self):
        """Test that a P-peak outside all RR intervals is marked with -1."""
        from p_detection import rr_for_p

        r_peaks = [100, 200, 300]
        p_peaks = [50, 150, 350]
        expected = [-1, 0, -1]

        result = rr_for_p(p_peaks, r_peaks) 
        self.assertEqual(result, expected)

    def test_extract_segments_normal_mode(self):
        """
        Test extract_segments in normal RR mode with synthetic ECG data.
        """
        from p_detection import extract_segments

        fs = 250
        duration = 10  # seconds
        t = np.linspace(0, duration, duration * fs)
        ecg = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))

        r_peaks = [250, 500, 750, 1000, 1250, 1500]
        rr_intervals = np.diff(r_peaks) / fs
        rr_intervals = rr_intervals.tolist()
        valid_beats = [rp for rp in r_peaks]  # All valid
        valid_beats[0] = None  # type: ignore # simulate invalid start
        begin = 0

        segments, segments_p, segments_t, win_val = extract_segments( # type: ignore
            ecg, rr_intervals, r_peaks, fs, valid_beats, begin, dis_p_check=False # type: ignore
        )

        self.assertEqual(len(segments), len(rr_intervals))
        self.assertEqual(len(segments_p), len(rr_intervals))
        self.assertEqual(len(segments_t), len(rr_intervals))
        self.assertIsInstance(segments[1], list)


    def test_extract_segments_dissociated_mode(self):
        """
        Test extract_segments in dissociated P-wave detection mode.
        """
        from p_detection import extract_segments

        fs = 250
        t = np.linspace(0, 10, 2500)
        ecg = np.sin(2 * np.pi * 1.0 * t)

        r_peaks = [250, 500, 750, 1000]
        rr_intervals = np.diff(r_peaks) / fs
        rr_intervals = rr_intervals.tolist()
        valid_beats = [1, 1, 1, 1]
        det_p = [725]
        det_t = [625]
        begin = 0

        dissociated_segs, t_vals = extract_segments( # type: ignore
            ecg, rr_intervals, r_peaks, fs, valid_beats, begin, # type: ignore
            dis_p_check=True, det_t=det_t, det_p=det_p
        )

        self.assertEqual(len(dissociated_segs), 1)
        self.assertTrue(all(isinstance(seg, list) for seg in dissociated_segs))


    def test_extract_segments_too_few_r_peaks(self):
        """
        Test extract_segments handles cases with not enough R-peaks to segment.
        """
        from p_detection import extract_segments

        fs = 250
        ecg = np.random.randn(1000)
        r_peaks = [100]  # Too few
        rr_intervals = []
        valid_beats = [None]
        begin = 0

        segments, segments_p, segments_t, win_val=extract_segments( # type: ignore
            ecg, rr_intervals, r_peaks, fs, valid_beats, begin) # type: ignore
        self.assertEqual(len(segments), 0)
        self.assertEqual(len(segments_p), 0)
        self.assertEqual(len(segments_t), 0)


    def test_extract_segments_all_invalid_beats(self):
        """
        Test extract_segments when all beats are invalid (None).
        """
        from p_detection import extract_segments

        fs = 250
        ecg = np.random.randn(1250)
        r_peaks = [250, 500, 750, 1000]
        rr_intervals = np.diff(r_peaks) / fs
        rr_intervals = rr_intervals.tolist()
        valid_beats = [None] * len(r_peaks)
        begin = 0

        segments, segments_p, segments_t, win_val = extract_segments( # type: ignore
            ecg, rr_intervals, r_peaks, fs, valid_beats, begin # type: ignore
        )

        self.assertTrue(all(seg == [] for seg in segments))


    def test_generate_template_basic_clustering(self):
        """
        Test generate_template clusters similar segments together.
        """
        from p_detection import generate_template
        np.random.seed(42)

        # Create 3 similar segments and 1 distinct one
        seg1 = np.sin(np.linspace(0, np.pi, 50)).tolist()
        seg2 = (np.sin(np.linspace(0, np.pi, 50)) +
                0.01 * np.random.randn(50)).tolist()
        seg3 = (np.sin(np.linspace(0, np.pi, 50)) +
                0.02 * np.random.randn(50)).tolist()
        seg4 = (np.cos(np.linspace(0, np.pi, 50))).tolist()

        segments = [seg1, seg2, seg3, seg4]
        templates, counts, scores, check_list, seg_map = generate_template(
            segments, pearson_threshold=0.95)

        # We expect 2 templates: 3 similar, 1 distinct
        self.assertEqual(len(templates), 2)
        self.assertEqual(sum(counts), 4)
        self.assertEqual(len(seg_map), 4)
        self.assertEqual(sorted(counts), [1, 3])  # one outlier


    def test_drop_low_weight_templates_behavior(self):
        """
        Test that low-weight templates are dropped and segments reassigned correctly.
        """
        from p_detection import drop_low_weight_templates
        import numpy as np

        np.random.seed(42)

        segments = [[1, 2, 3], [4, 5, 6], [1, 1, 1],
                    [2, 2, 2], [7, 8, 9], [3, 3, 3]]
        templates = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([1, 1, 1]),
                    np.array([7, 8, 9]), np.array([3, 3, 3])]

        # Mixed counts: some below threshold, some exactly 1, and some sufficient.
        counts = [1, 15, 1, 9, 1]
        seg_map = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 4}
        qrs_indices = list(range(100))
        perce = 0.05  # 5% threshold

        updated_templates, updated_counts, updated_map, updated_segments = drop_low_weight_templates(
            segments, templates, counts, seg_map, qrs_indices, perce # type: ignore
        )

        # Templates with sufficient counts (>=10) or within 5 templates total should be preserved
        self.assertEqual(len(updated_templates), 5)
        self.assertEqual(updated_counts, counts)


    def test_generate_template_empty_input(self):
        """
        Ensure generate_template handles empty input gracefully.
        """
        from p_detection import generate_template

        templates, counts, scores, check_list, seg_map = generate_template([])
        self.assertEqual(templates, [])
        self.assertEqual(counts, [])
        self.assertEqual(scores, [])
        self.assertEqual(check_list, [])
        self.assertEqual(seg_map, {})


    def test_safe_savgol_filter_short_signal(self):
        """
        Ensure safe_savgol_filter returns original signal if too short.
        """
        from p_detection import safe_savgol_filter
        x = np.array([1, 2, 3])  # Too short for window 10
        smoothed = safe_savgol_filter(x, window_length=10, polyorder=3)
        np.testing.assert_array_equal(smoothed, x)


    def test_find_best_peak_new_returns_valid_index(self):
        """
        Test that find_best_peak_new returns a valid peak index on synthetic phasor signals.
        """
        from p_detection import find_best_peak

        fs = 250
        t = np.linspace(0, 1, fs)
        template = np.sin(2 * np.pi * 5 * t)
        new = np.abs(template)
        new_minus = -template
        new_tem = new.copy()
        tem = new.copy()
        tem_min = new_minus.copy()

        peak = find_best_peak(new, new_minus, new_tem,
                                tem, tem_min, template, plot=False)
        self.assertTrue(isinstance(peak, (int, np.integer)) or peak is None)
        if peak is not None:
            self.assertTrue(0 <= peak < len(template))


    def test_detect_peaks_in_templates_valid_output(self):
        """
        Test detect_peaks_in_templates returns valid indices for basic templates.
        """
        from p_detection import detect_peaks_in_templates
        fs = 250
        t = np.linspace(0, 1, fs)

        # Create two synthetic templates with known peaks
        template1 = np.sin(2 * np.pi * 5 * t)
        template2 = np.sin(2 * np.pi * 7 * t)
        templates = [template1, template2]

        peaks = detect_peaks_in_templates(templates, plot=False)
        self.assertEqual(len(peaks), len(templates))
        for p in peaks:
            self.assertTrue((p is None) or (0 <= p < fs))


    def test_map_peaks_to_real_segments_mapping(self):
        """
        Test mapping from template peak to real ECG index.
        """
        from p_detection import map_peaks_to_real_segments

        fs = 250
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, fs * 2))
        qrs_indices = [100, 200, 300, 400]
        rr_int = [(qrs_indices[i+1] - qrs_indices[i]) /
                fs for i in range(len(qrs_indices)-1)]
        segments = [signal[100:150], signal[200:250], signal[300:350]]
        segments_all = segments.copy()
        templates = [np.linspace(0, 1, 50), np.linspace(0, 1, 50)]
        peak_indices = [25, 30]
        segment_to_template = {0: 0, 1: 1, 2: 1}
        begin = 0

        mapped_indices = map_peaks_to_real_segments(
            signal, templates, segments_all, segments,
            segment_to_template, peak_indices, # type: ignore
            p=True, dis=False,
            qrs_indices=qrs_indices, rr_int=rr_int, begin=begin
        )

        self.assertIsInstance(mapped_indices, list)
        self.assertTrue(all(isinstance(x, (int, np.integer))
                         for x in mapped_indices))


    def test_predict_dis_p_waves_int_logic(self):
        """
        Ensure dissociation logic matches RR interval jumps and PVC flags.
        """
        from p_detection import predict_dis_p_waves_int

        rr = [0.8, 0.85, 1.8, 1.7, 2.0]
        r_peaks = [100, 200, 300, 400, 500, 600]
        p_candidates = [110, 210, 320, 430, 550]
        pvc_labels = [False, False, False, True, False]
        signal = np.random.randn(1000)
        fs = 250
        begin = 0

        result = predict_dis_p_waves_int(rr, pvc_labels)

        self.assertEqual(len(result), len(rr))
        self.assertIsInstance(result[0], bool)

    """
    def test_p_dis_temp_gen_new_runs(self):

        from p_detection import p_dis_temp_gen

        # Fake inputs
        fs = 250
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 2, fs * 2))
        qrs_indices = [100, 200, 300, 400]
        rr_int = [(qrs_indices[i+1] - qrs_indices[i]) /
                fs for i in range(len(qrs_indices)-1)]
        segments = [signal[100:150], signal[200:250], signal[300:350]]
        segments_dis = [signal[120:140], signal[220:240], signal[320:340]]
        det_t = [150, 250, 350]
        det_p = [145, 245, 340]
        p_dis = [False, True, True]
        begin = 0

        indices = p_dis_temp_gen(
            p_dis, segments, segments_dis, qrs_indices,
            signal, rr_int, det_t, det_p, begin, seg=False, plot=False
        )

        self.assertIsInstance(indices, list)
        if indices:
            self.assertTrue(all(isinstance(x, int) for x in indices))"""


    
    def setUp(self):
        # Generate synthetic signal (sinusoidal + noise)
        self.fs = 250
        self.duration_sec = 10
        self.begin = 0
        self.samples = self.fs * self.duration_sec
        t = np.linspace(0, self.duration_sec, self.samples)
        signal = 0.3 * np.sin(2 * np.pi * 1.0 * t) + \
            0.05 * np.random.randn(self.samples)

        # Simulate R-peaks roughly every second
        self.qrs_indices = [i for i in range(100, self.samples, 250)]
        self.pvc_indices = [self.qrs_indices[3]]  # Simulate one PVC
        self.afib_indices = []  # No AFIB in this test case

        self.signal = signal
        self.indices = [self.qrs_indices, self.pvc_indices, self.afib_indices]

    def test_algorithm_runs(self):
        from p_detection import algorithm
        try:
            p_peaks, t_peaks = algorithm(
                signal=self.signal,
                indices=self.indices,
                begin=self.begin,
                fs=self.fs,
                seg=False,
                p_co=0.8,
                plot=False,
                pt_v=0.1
            )
            # Ensure outputs are lists of integers or floats
            self.assertIsInstance(p_peaks, list)
            self.assertIsInstance(t_peaks, list)
            for p in p_peaks:
                self.assertIsInstance(p, (int, np.integer))
            for t in t_peaks:
                self.assertIsInstance(t, (int, np.integer))
        except Exception as e:
            self.fail(f"Algorithm execution failed with error: {e}")




if __name__ == '__main__':
    unittest.main()
