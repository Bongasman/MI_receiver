import serial
import struct
import time
import threading
import queue
import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, lfilter, spectrogram, firwin
from scipy.signal.windows import tukey

# ============================================================
# CONFIGURATION – tweak here
# ============================================================

# --- I/O / ADC ---
PORT = 'COM9'
BAUD_RATE = 921600
SAMPLE_SIZE = 2          # bytes per sample (uint16)
ADC_FS = 16000           # hardware ADC sampling rate [Hz]
PROC_FS = 8000           # processing sample rate after downsampling
FS = PROC_FS             # legacy alias used throughout DSP path

# --- Downsampling / processing rate ---
if PROC_FS > ADC_FS:
    raise ValueError("PROC_FS cannot exceed ADC_FS")
USE_DOWNSAMPLING = (PROC_FS != ADC_FS)
if USE_DOWNSAMPLING:
    if ADC_FS % PROC_FS != 0:
        raise ValueError("ADC_FS must be an integer multiple of PROC_FS")
    DECIMATION_FACTOR = ADC_FS // PROC_FS
    DOWNSAMPLE_FILTER_TAPS = 129
    # keep comfortable guard from Nyquist to avoid aliasing artifacts
    DOWNSAMPLE_CUTOFF_HZ = max(1000.0, min(PROC_FS / 2 - 200.0, PROC_FS * 0.45))
else:
    DECIMATION_FACTOR = 1
    DOWNSAMPLE_FILTER_TAPS = 0
    DOWNSAMPLE_CUTOFF_HZ = None

# --- LFM (chirp) parameters (must match TX) ---
LFM_TSYMB_SEC = 1.024    # chirp duration [s]
LFM_BW_HZ = 100.0        # sweep bandwidth [Hz]
LFM_REPEATS = 1          # repeats (L)
LFM_FC_HZ = 950.0        # mixing (carrier) frequency [Hz]

# --- BFSK parameters ---
BFSK_CENTER_HZ = 950.0   # center freq of BFSK [Hz]
BFSK_DEV_HZ = 50.0       # deviation around center [Hz]
BFSK_SYMBOL_SEC = 0.02    # symbol duration [s] (derived from 320 @16 kHz)
OVS = int(round(BFSK_SYMBOL_SEC * FS))
USE_HANNING_DEMOD = True # apply Hann per symbol in demod correlator
USE_SPECTROGRAM_DEMOD = False  # when True, use spectrogram-based BFSK demod instead of time-domain correlators
SPECTROGRAM_DEMOD_BAND_HZ = 60.0  # +/- band (Hz) summed around each tone when using spectrogram demod
SPECTROGRAM_WINDOW = 'hann'

# --- Band-pass filter for spectrogram / demod band ---
USE_BANDPASS = False     # <--- toggle this to enable/disable BPF
BPF_LOW_HZ = 800.0      # lower cutoff [Hz]
BPF_HIGH_HZ = 1200.0    # upper cutoff [Hz]
BPF_ORDER = 4

# --- Detection / recording window ---
MIN_PEAK_CORR = 1.4    # threshold on |corr| (tune based on data)
GUARD_SEC = 1         # guard interval after LFM end [s]
GUARD_SAMPLES = int(round(FS * GUARD_SEC))

RECORD_DURATION_SEC = 44160 / 16000.0  # maintain ~2.76 s capture window
RECORD_SAMPLES = int(round(RECORD_DURATION_SEC * FS))

# --- Normalized-correlation / factor thresholds ---
# Tune these manually by watching the [CORR] log output.
NORM_PEAK_THRESH = 0.25   # absolute threshold on normalized correlation magnitude (0..1)
FACTOR_THRESH = 8.0       # peak / mean_excluding_peak ratio on normalized corr

# --- Correlation settings ---
CORR_WINDOW_SEC = 3.0                 # seconds of history used for LFM correlation    -------------------
CORR_HOP_SAMPLES = FS // 10           # run correlation every ~0.1 s of new data

# --- Spectrogram settings (MATLAB-like) ---
PLOT_WINDOW_SEC = 2.0                 # seconds visible in spectrogram
NFFT = 256
SPEC_OVERLAP_FRACTION = 0.25          # 0.5 to 0.75 typical
NOVERLAP = int(NFFT * SPEC_OVERLAP_FRACTION)
SHOW_DECODED_ON_PLOT = True          # overlay latest decoded values on spectrogram when True
SPEC_VMIN_DB = -110.0                 # lower limit for spectrogram color scale (dB)
SPEC_VMAX_DB = -40.0                  # upper limit for spectrogram color scale (dB)

# --- Results file ---
RESULTS_FILE = "decoded_results.txt"

# --- Sensor validation thresholds ---
TEMP_MIN_VALID = 10.0
TEMP_MAX_VALID = 50.0
TEMP_FALLBACK_VALUE = 20.2

# --- Diagnostic periodic save ---
DIAG_PERIODIC_SAVE = False    # set True to write a snapshot every DIAG_SAVE_INTERVAL_SEC
DIAG_SAVE_INTERVAL_SEC = 30  # seconds between diagnostic snapshots
DIAG_SAVE_DIR = "diagnostics"

# --- Runtime mode flags ---
DISABLE_CORRELATION = False   # when True, don't run LFM cross-correlation
DISABLE_DEMOD = False        # when True, don't run demodulation / viterbi on recorded blocks

# --- Full 30s recording buffer ---
RECORD_FULL_30SEC = False    # when True, periodically write full 30s captures instead of small snapshots
FULL_REC_SEC = 30
FULL_REC_SAMPLES = int(FS * FULL_REC_SEC)

# ============================================================
# GLOBALS
# ============================================================

sample_queue = queue.Queue()
running = True
writer_queue = queue.Queue()
writer_running = True

# pending absolute start index for the next recorded FSK block (set at detection)
pending_fsk_abs_start = None

# Absolute sample counter (incremented as samples are consumed) — helps map
# buffer indices to an absolute stream sample number for debugging and logging.
sample_counter = 0

# Recording state (after detection)
recording_active = False
record_samples_remaining = 0
record_buffer = []      # list[(idx, adc_val)]
decode_index = 1        # detection counter
record_index = 0        # index inside current recording (0..RECORD_SAMPLES-1)
latest_decoded_text = ""
latest_temperature = None
latest_conductivity = None
latest_depth = None

if USE_DOWNSAMPLING and DECIMATION_FACTOR > 1:
    nyq_adc = 0.5 * ADC_FS
    cutoff_norm = min(0.99, max(1e-3, DOWNSAMPLE_CUTOFF_HZ / nyq_adc))
    decim_b = firwin(DOWNSAMPLE_FILTER_TAPS, cutoff_norm, window='hann').astype(np.float32)
    decim_zi = np.zeros(len(decim_b) - 1, dtype=np.float32)
    decim_phase = 0
else:
    decim_b = None
    decim_zi = None
    decim_phase = 0


def apply_downsampling(x: np.ndarray) -> np.ndarray:
    """Downsample the incoming float32 chunk to PROC_FS using the FIR decimator."""
    global decim_zi, decim_phase
    if decim_b is None or DECIMATION_FACTOR == 1:
        return x
    y, decim_zi = lfilter(decim_b, [1.0], x, zi=decim_zi)
    start = decim_phase % DECIMATION_FACTOR
    down = y[start::DECIMATION_FACTOR]
    decim_phase = (start + len(y)) % DECIMATION_FACTOR
    return down


# ============================================================
# UART READER THREAD
# ============================================================

def serial_reader():
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"[INFO] Reading from {PORT} at {BAUD_RATE} baud...")
    leftover = b''

    while running:
        try:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            data = leftover + data
            n = len(data) // SAMPLE_SIZE * SAMPLE_SIZE
            valid = data[:n]
            leftover = data[n:]
            for i in range(0, len(valid), SAMPLE_SIZE):
                # ADC sends unsigned 16-bit words; convert to signed int16 here
                val_u = struct.unpack('<H', valid[i:i+2])[0]
                if val_u >= 0x8000:
                    val = val_u - 0x10000
                else:
                    val = val_u
                sample_queue.put(val)
        except Exception as e:
            print(f"[ERROR] Serial read failed: {e}")
            break

    ser.close()
    print("[UART] Serial reader stopped.")


# ============================================================
# LFM REFERENCE GENERATOR (Python version of your MATLAB code)
# ============================================================

def generate_lfm_ref(Fs, Tsymb=LFM_TSYMB_SEC, B=LFM_BW_HZ, L=LFM_REPEATS, fc=LFM_FC_HZ):
    """
    Generate real-valued LFM reference, equivalent to your MATLAB:
      - complex baseband LFM: phi(t) = pi * B * t * (t/Tsymb - 1)
      - Tukey window alpha=0.1
      - repeat L times
      - mix with exp(-j*2*pi*fc*t), take real part
      - remove DC
      - L2-normalize for correlation
    """
    M = int(round(Tsymb * Fs))  # samples per chirp
    i = np.arange(M, dtype=np.float32)
    t = i * Tsymb / M           # matches C: i * Tsymb / M

    phi = np.pi * B * t * (t / Tsymb - 1.0)
    LFM_symbol = np.exp(1j * phi)

    w = tukey(M, alpha=0.1)
    LFM_symbol *= w

    lfm_complex = np.tile(LFM_symbol, L)

    N = len(lfm_complex)
    n = np.arange(N, dtype=np.float32)
    t2 = n / Fs
    carrier = np.exp(-1j * 2.0 * np.pi * fc * t2)
    mixed = lfm_complex * carrier
    y = np.real(mixed).astype(np.float32)

    y -= np.mean(y)

    norm = np.linalg.norm(y)
    if norm > 0:
        y /= norm

    return y


# ============================================================
# BFSK DEMOD / VITERBI / FIXED-Q8.7
# ============================================================

F0 = BFSK_CENTER_HZ - BFSK_DEV_HZ
F1 = BFSK_CENTER_HZ + BFSK_DEV_HZ

K = 9
NUM_STATES = 1 << (K - 1)
NUM_OUTPUTS = 2
G = [[1, 1, 1, 1, 0, 1, 0, 1, 1],
     [1, 0, 1, 1, 1, 0, 0, 0, 1]]


def demodulate_bfsk(rx_data):
    """
    BFSK demod: tones at F0 and F1, symbol length = OVS samples.
    rx_data: 1D float array at FS (already DC-removed, band-limited if desired).
    """
    t = np.arange(OVS) / FS
    ref0 = np.cos(2 * np.pi * F0 * t)
    ref1 = np.cos(2 * np.pi * F1 * t)

    if USE_HANNING_DEMOD:
        w = np.hanning(OVS)
        ref0 *= w
        ref1 *= w

    num_symbols = len(rx_data) // OVS
    bits = []

    for s in range(num_symbols):
        symbol = rx_data[s * OVS:(s + 1) * OVS]
        symbol *= w # instead of windowing refs – equivalent)
        corr0 = np.dot(symbol, ref0)
        corr1 = np.dot(symbol, ref1)
        bits.append(0 if abs(corr0) > abs(corr1) else 1)

    return bits


def demodulate_bfsk_spectrogram(rx_data):
    """
    Alternative BFSK demodulator that works in the frequency domain using
    an STFT/spectrogram. Each symbol-length slice (OVS samples) is windowed,
    transformed, and the energy near F0 vs F1 is compared.
    """
    if len(rx_data) < OVS:
        return []

    # Compute spectrogram with one column per symbol (nperseg=OVS, no overlap).
    freqs, times, Sxx = spectrogram(
        rx_data,
        fs=FS,
        window=SPECTROGRAM_WINDOW,
        nperseg=OVS,
        noverlap=0,
        nfft=OVS,
        scaling='spectrum',
        mode='magnitude',
    )

    if Sxx.size == 0:
        return []

    def _band_slice(target_hz):
        if len(freqs) == 0:
            return slice(0, 0)
        bin_res = freqs[1] - freqs[0] if len(freqs) > 1 else FS / OVS
        half_bins = max(0, int(np.ceil(SPECTROGRAM_DEMOD_BAND_HZ / max(bin_res, 1e-9))))
        idx_center = int(np.argmin(np.abs(freqs - target_hz)))
        start = max(0, idx_center - half_bins)
        stop = min(len(freqs), idx_center + half_bins + 1)
        return slice(start, stop)

    band0 = _band_slice(F0)
    band1 = _band_slice(F1)

    bits = []
    # Iterate over each symbol column (second axis of Sxx is time/segment).
    for col in Sxx.T:
        p0 = np.sum(col[band0])
        p1 = np.sum(col[band1])
        bits.append(0 if p0 >= p1 else 1)

    return bits


def hamming_distance(a, b):
    return bin(a ^ b).count('1')


def viterbi_decode(encoded_bits):
    num_steps = len(encoded_bits) // NUM_OUTPUTS
    if num_steps == 0:
        return []

    path_metrics = [float('inf')] * NUM_STATES
    path_metrics[0] = 0
    paths = [[0] * num_steps for _ in range(NUM_STATES)]

    for step in range(num_steps):
        new_pm = [float('inf')] * NUM_STATES
        new_paths = [[0] * num_steps for _ in range(NUM_STATES)]
        for state in range(NUM_STATES):
            if path_metrics[state] == float('inf'):
                continue
            for bit in [0, 1]:
                new_state = ((state << 1) | bit) & (NUM_STATES - 1)
                output = 0
                for i_out in range(NUM_OUTPUTS):
                    parity = bit
                    for j in range(K - 1):
                        parity ^= ((state >> j) & 1) * G[i_out][j + 1]
                    output = (output << 1) | parity
                recv = (encoded_bits[2 * step] << 1) | encoded_bits[2 * step + 1]
                metric = path_metrics[state] + hamming_distance(output, recv)
                if metric < new_pm[new_state]:
                    new_pm[new_state] = metric
                    new_paths[new_state][:step] = paths[state][:step]
                    new_paths[new_state][step] = bit
        path_metrics = new_pm
        paths = new_paths

    return paths[np.argmin(path_metrics)]


def fixed_q8_7_to_float(bits):
    assert len(bits) == 16
    val = sum((bits[i] << (15 - i)) for i in range(16))
    if val >= (1 << 15):
        val -= (1 << 16)
    return val / 128.0


def apply_bounded_update(
    label,
    current_value,
    new_value,
    *,
    min_value=None,
    min_inclusive=True,
    max_value=None,
    max_inclusive=True,
):
    """Update only when the new reading satisfies all configured bounds."""

    if min_value is not None:
        if min_inclusive and new_value < min_value:
            print(
                f"[DEMOD] {label} reading {new_value:.6f} is below minimum {min_value:.3f}; retaining previous value."
            )
            return current_value, False
        if not min_inclusive and new_value <= min_value:
            print(
                f"[DEMOD] {label} reading {new_value:.6f} must be > {min_value:.3f}; retaining previous value."
            )
            return current_value, False

    if max_value is not None:
        if max_inclusive and new_value > max_value:
            print(
                f"[DEMOD] {label} reading {new_value:.6f} exceeds maximum {max_value:.3f}; retaining previous value."
            )
            return current_value, False
        if not max_inclusive and new_value >= max_value:
            print(
                f"[DEMOD] {label} reading {new_value:.6f} must be < {max_value:.3f}; retaining previous value."
            )
            return current_value, False

    return new_value, True


def append_decoded_result(index, temperature, conductivity, pressure):
    timestamp = datetime.now().isoformat()
    line = f"{timestamp},{index},{temperature:.6f},{conductivity:.6f},{pressure:.6f}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(line)


def process_recorded_block(record_buffer, detection_index):
    """
    Take the RECORD_SAMPLES block (raw ADC counts), demodulate BFSK,
    Viterbi decode, parse Q8.7 values, and print/log the results.
    record_buffer: list[(idx, adc_val)]
    """
    global latest_decoded_text, latest_temperature, latest_conductivity, latest_depth
    if not record_buffer:
        print("[DEMOD] Empty record_buffer, nothing to decode.")
        return

    raw_vals = np.array([v for (_, v) in record_buffer], dtype=np.float32)

    # raw_vals are signed int16 in [-32768, 32767]
    x = raw_vals / 32768.0
    x -= np.mean(x)

    if USE_SPECTROGRAM_DEMOD:
        bits_rx = demodulate_bfsk_spectrogram(x)
        demod_method = "spectrogram"
    else:
        bits_rx = demodulate_bfsk(x)
        demod_method = "correlator"

    print(f"[DEMOD] Using {demod_method} demodulator; obtained {len(bits_rx)} bits.")
    if not bits_rx:
        print("[DEMOD] No bits demodulated from recorded block.")
        return

    decoded_bits = viterbi_decode(bits_rx)
    if len(decoded_bits) < 56:
        print(f"[DEMOD] Not enough decoded bits ({len(decoded_bits)}) for 3x Q8.7 values.")
        return

    sync_bits = decoded_bits[:8]
    if sync_bits != [1] * 8:
        print(f"[DEMOD] Sync not detected. First 8 decoded bits: {sync_bits}")
    else:
        print("[DEMOD] Sync word detected (8 ones).")

    temp_bits = decoded_bits[8:24]
    cond_bits = decoded_bits[24:40]
    pres_bits = decoded_bits[40:56]

    decoded_temp = fixed_q8_7_to_float(temp_bits)
    decoded_cond = 0.0
    decoded_pres = 0.0

    print(f"[DEMOD] Detection #{detection_index}")
    print(f"        Temperature:  {decoded_temp:.6f}")
    print(f"        Conductivity: {decoded_cond:.6f} µS/cm")
    print(f"        Depth:     {decoded_pres:.6f} m")

    latest_temperature, temp_updated = apply_bounded_update(
        "Temperature",
        latest_temperature,
        decoded_temp,
        min_value=TEMP_MIN_VALID,
        min_inclusive=False,
        max_value=TEMP_MAX_VALID,
        max_inclusive=False,
    )
    if not temp_updated:
        print(
            f"[DEMOD] Temperature out of range; forcing fallback {TEMP_FALLBACK_VALUE:.1f} °C"
        )
        latest_temperature = TEMP_FALLBACK_VALUE
    latest_conductivity, _ = apply_bounded_update(
        "Conductivity",
        latest_conductivity,
        decoded_cond,
        min_value=0.0,
        min_inclusive=True,
    )
    latest_depth, _ = apply_bounded_update(
        "Depth",
        latest_depth,
        decoded_pres,
        min_value=0.0,
        min_inclusive=True,
    )

    if latest_temperature is not None and latest_conductivity is not None and latest_depth is not None:
        append_decoded_result(detection_index, latest_temperature, latest_conductivity, latest_depth)
    else:
        print("[DEMOD] Skipping results file update until all sensor readings have valid (>= 0) values.")

    def _format_value(value, suffix):
        if value is None:
            return f"N/A {suffix}".strip()
        return f"{value:.2f} {suffix}".strip()

    latest_decoded_text = (
        # f"Det\n"
        # f"T={_format_value(latest_temperature, '°C')}\n"
        # f"Cond={_format_value(latest_conductivity, 'µS/cm')}\n"
        # f"Depth={_format_value(latest_depth, 'm')}"
    )


# ============================================================
# MAIN LOOP: SPECTROGRAM + CORRELATION + DEMOD/DECODE
# ============================================================

def main(lfm_normalized=None):
    global recording_active, record_samples_remaining, record_buffer
    global decode_index, record_index

    # --- Spectrogram setup ---
    plot_window_samples = int(PLOT_WINDOW_SEC * FS)
    plot_buf = np.zeros(plot_window_samples, dtype=np.float32)
    buf_filled = 0

    spec_window = np.hanning(NFFT)
    cb = None  # colorbar handle
    decoded_text_artist = None

    # --- BPF design for spectrogram band ---
    # --- BPF design (optional) for spectrogram band ---
    if USE_BANDPASS:
        nyq = 0.5 * FS
        b_bp, a_bp = butter(
            BPF_ORDER,
            [BPF_LOW_HZ / nyq, BPF_HIGH_HZ / nyq],
            btype="band"
        )
        zi_bp = np.zeros(max(len(a_bp), len(b_bp)) - 1, dtype=np.float32)
    else:
        b_bp = a_bp = None
        zi_bp = None

    # If a reference exists and bandpass is enabled, filter the reference
    # with the same (causal) BPF used in the real-time path, then remove
    # DC and renormalize so correlation remains consistent.
    if lfm_normalized is not None and USE_BANDPASS and b_bp is not None:
        try:
            lfm_normalized = lfilter(b_bp, a_bp, lfm_normalized)
            lfm_normalized = lfm_normalized - np.mean(lfm_normalized)
            norm = np.linalg.norm(lfm_normalized)
            if norm > 0:
                lfm_normalized = lfm_normalized / norm
        except Exception as e:
            print(f"[WARN] Failed to filter LFM reference with BPF: {e}")


    # --- Correlation + raw buffers ---
    if lfm_normalized is not None:
        LFM_LEN = len(lfm_normalized)

        corr_window_samples = max(int(CORR_WINDOW_SEC * FS), LFM_LEN + 1)
        corr_buf = np.zeros(corr_window_samples, dtype=np.float32)
        corr_filled = 0
        samples_since_last_corr = 0

        raw_window_samples = corr_window_samples
        raw_buf = np.zeros(raw_window_samples, dtype=np.int32)
        raw_filled = 0
    else:
        LFM_LEN = 0
        corr_buf = None
        corr_filled = 0
        samples_since_last_corr = 0
        raw_buf = None
        raw_filled = 0
        raw_window_samples = 0
    # full 30s circular buffer for periodic full-record snapshots (exists regardless)
    full_buf = np.zeros(FULL_REC_SAMPLES, dtype=np.int32)
    full_write_ptr = 0
    full_filled = 0

    # --- Matplotlib interactive setup ---
    plt.ion()
    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title("Real-time Spectrogram")
    except Exception:
        pass

    # --- Diagnostic periodic save state ---
    # Ensure diagnostics folder exists and determine next diag_counter regardless
    os.makedirs(DIAG_SAVE_DIR, exist_ok=True)
    last_diag_save_ts = time.time()
    existing = glob.glob(os.path.join(DIAG_SAVE_DIR, "adc_*.txt"))
    max_idx = 0
    for path in existing:
        bn = os.path.basename(path)
        m = re.search(r'_(\d+)\.txt$', bn)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
    diag_counter = max_idx + 1
    # determine next det_counter for saved FSK blocks (det_1, det_2, ...)
    existing_det = glob.glob(os.path.join(DIAG_SAVE_DIR, "det_*.txt"))
    max_det = 0
    for path in existing_det:
        bn = os.path.basename(path)
        m = re.search(r'_(\d+)\.txt$', bn)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_det:
                    max_det = idx
            except Exception:
                pass
    det_counter = max_det + 1

    # --- Background writer thread ---
    def writer_worker():
        global writer_running
        while writer_running:
            try:
                task = writer_queue.get(timeout=0.5)
            except Exception:
                continue
            try:
                fname, snapshot, abs_start, fs = task
                # write two-column file: global_index value (no timestamp, no header)
                n = len(snapshot)
                if n == 0:
                    # create empty file
                    open(fname, 'w').close()
                else:
                    indices = np.arange(int(abs_start), int(abs_start) + n, dtype=np.int64)
                    if snapshot.dtype == np.int32 or np.issubdtype(snapshot.dtype, np.integer):
                        out = np.column_stack((indices, snapshot.astype(np.int64)))
                        np.savetxt(fname, out, fmt='%d %d')
                    else:
                        out = np.column_stack((indices, snapshot.astype(np.float64)))
                        np.savetxt(fname, out, fmt='%d %.6f')
                print(f"[WRITER] Wrote {fname} (samples={n})")
            except Exception as e:
                print(f"[WRITER] Error writing file: {e}")
            finally:
                try:
                    writer_queue.task_done()
                except Exception:
                    pass

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    # --- Main loop ---
    while running:
        try:
            raw_vals_list = []
            while not sample_queue.empty():
                val = sample_queue.get()
                raw_vals_list.append(val)

            if raw_vals_list:
                raw_vals_int = np.array(raw_vals_list, dtype=np.int32)
                if raw_vals_int.size == 0:
                    continue

                x_adc = raw_vals_int.astype(np.float32) / 32768.0
                x_adc -= np.mean(x_adc)

                proc_vals = apply_downsampling(x_adc)
                proc_vals = np.asarray(proc_vals, dtype=np.float32)
                if proc_vals.size == 0:
                    continue

                n_in = len(proc_vals)
                raw_chunk = np.clip(np.round(proc_vals * 32768.0), -32768, 32767).astype(np.int32)

                # update absolute sample counter (total samples consumed so far)
                # we increment here so the 'now' mapping (sample_counter) includes
                # the samples just pushed into the rolling buffers below.
                global sample_counter
                sample_counter += n_in

                # write into full 30s circular buffer (if enabled)
                if RECORD_FULL_30SEC and full_buf is not None:
                    raw_chunk_full = raw_chunk
                    n_raw_full = len(raw_chunk_full)
                    if n_raw_full >= FULL_REC_SAMPLES:
                        # take last FULL_REC_SAMPLES samples
                        full_buf[:] = raw_chunk_full[-FULL_REC_SAMPLES:]
                        full_filled = FULL_REC_SAMPLES
                        full_write_ptr = 0
                    else:
                        end = full_write_ptr + n_raw_full
                        if end <= FULL_REC_SAMPLES:
                            full_buf[full_write_ptr:end] = raw_chunk_full
                            full_write_ptr = end % FULL_REC_SAMPLES
                        else:
                            first = FULL_REC_SAMPLES - full_write_ptr
                            full_buf[full_write_ptr:] = raw_chunk_full[:first]
                            rem = n_raw_full - first
                            full_buf[:rem] = raw_chunk_full[first:]
                            full_write_ptr = rem
                        full_filled = min(FULL_REC_SAMPLES, full_filled + n_raw_full)

                # 0) continue a running recording if active
                if recording_active and record_samples_remaining > 0:
                    n_cap = min(record_samples_remaining, n_in)
                    for i in range(n_cap):
                        sample_index = record_index
                        record_buffer.append((int(sample_index),
                                              int(raw_chunk[i])))
                        record_index += 1
                    record_samples_remaining -= n_cap

                    if record_samples_remaining == 0:
                            print(f"[REC] Collected full {RECORD_SAMPLES} samples for detection #{decode_index}.")

                            # assemble numeric snapshot from record_buffer (values only)
                            vals = np.array([v for (_, v) in record_buffer], dtype=np.int32)

                            # enqueue write of aligned recorded block using pending_fsk_abs_start
                            try:
                                abs_start = int(pending_fsk_abs_start) if pending_fsk_abs_start is not None else (sample_counter - len(vals))
                            except Exception:
                                abs_start = sample_counter - len(vals)

                            fname = os.path.join(DIAG_SAVE_DIR, f"det_{det_counter}.txt")
                            det_counter += 1
                            writer_queue.put((fname, vals, abs_start, FS))
                            print(f"[WRITER] Enqueued recorded block write: {fname} (abs_start={abs_start}, samples={len(vals)})")

                            if not DISABLE_DEMOD:
                                process_recorded_block(record_buffer, decode_index)
                            else:
                                print("[DEMOD] Demodulation disabled; skipping processing of recorded block.")

                            decode_index += 1
                            recording_active = False
                            record_buffer = []
                            record_index = 0

                # A) update raw history buffer
                if raw_buf is not None:
                    n_raw_new = len(raw_chunk)

                    if raw_filled < raw_window_samples:
                        space = raw_window_samples - raw_filled
                        if n_raw_new <= space:
                            raw_buf[raw_filled:raw_filled + n_raw_new] = raw_chunk
                            raw_filled += n_raw_new
                        else:
                            raw_buf[raw_filled:] = raw_chunk[:space]
                            raw_filled = raw_window_samples
                            extra = n_raw_new - space
                            if extra > 0:
                                raw_buf[:-extra] = raw_buf[extra:]
                                raw_buf[-extra:] = raw_chunk[-extra:]
                    else:
                        if n_raw_new >= raw_window_samples:
                            raw_buf[:] = raw_chunk[-raw_window_samples:]
                            raw_filled = raw_window_samples
                        else:
                            raw_buf[:-n_raw_new] = raw_buf[n_raw_new:]
                            raw_buf[-n_raw_new:] = raw_chunk
                            raw_filled = raw_window_samples

                # convert to float, center, remove DC (operating at PROC_FS)
                x = proc_vals.copy()
                x -= np.mean(x)

                # 1) BPF for spectrogram band (apply only if configured)
                if b_bp is not None:
                    y_bp, zi_bp = lfilter(b_bp, a_bp, x, zi=zi_bp)
                else:
                    # no bandpass configured — use raw signal for spectrogram/correlation
                    y_bp = x

                n_new = len(y_bp)
                if buf_filled < plot_window_samples:
                    space = plot_window_samples - buf_filled
                    if n_new <= space:
                        plot_buf[buf_filled:buf_filled + n_new] = y_bp
                        buf_filled += n_new
                    else:
                        plot_buf[buf_filled:] = y_bp[:space]
                        buf_filled = plot_window_samples
                        extra = n_new - space
                        if extra > 0:
                            plot_buf[:-extra] = plot_buf[extra:]
                            plot_buf[-extra:] = y_bp[-extra:]
                else:
                    if n_new >= plot_window_samples:
                        plot_buf[:] = y_bp[-plot_window_samples:]
                    else:
                        plot_buf[:-n_new] = plot_buf[n_new:]
                        plot_buf[-n_new:] = y_bp

                # 2) LFM cross-correlation on the chosen input (bandpassed if enabled)
                if (not DISABLE_CORRELATION) and lfm_normalized is not None and LFM_LEN > 0:
                    corr_input = y_bp
                    n_corr_new = len(corr_input)

                    if corr_filled < corr_window_samples:
                        space = corr_window_samples - corr_filled
                        if n_corr_new <= space:
                            corr_buf[corr_filled:corr_filled + n_corr_new] = corr_input
                            corr_filled += n_corr_new
                        else:
                            corr_buf[corr_filled:] = corr_input[:space]
                            corr_filled = corr_window_samples
                            extra = n_corr_new - space
                            if extra > 0:
                                corr_buf[:-extra] = corr_buf[extra:]
                                corr_buf[-extra:] = corr_input[-extra:]
                    else:
                        if n_corr_new >= corr_window_samples:
                            corr_buf[:] = corr_input[-corr_window_samples:]
                            corr_filled = corr_window_samples
                        else:
                            corr_buf[:-n_corr_new] = corr_buf[n_corr_new:]
                            corr_buf[-n_corr_new:] = corr_input
                            corr_filled = corr_window_samples

                    samples_since_last_corr += n_corr_new

                    if corr_filled >= LFM_LEN and samples_since_last_corr >= CORR_HOP_SAMPLES:
                        segment = corr_buf[:corr_filled]

                        # raw linear correlation (valid outputs)
                        corr = correlate(segment, lfm_normalized, mode="valid")

                        if corr.size > 0:
                            # compute running window L2 norms via convolution of squared signal
                            seg_sq = segment * segment
                            window = np.ones(LFM_LEN, dtype=np.float32)
                            sumsq = np.convolve(seg_sq, window, mode='valid')
                            local_norm = np.sqrt(np.maximum(sumsq, 1e-12))

                            ref_norm = np.linalg.norm(lfm_normalized)
                            corr_norm = corr / (local_norm * ref_norm + 1e-12)
                            corr_mag = np.abs(corr_norm)

                            max_idx_local = int(np.argmax(corr_mag))
                            peak_val = float(corr_mag[max_idx_local])
                            corr_index = max_idx_local

                            # mean excluding the peak (simple exclusion)
                            if corr_mag.size > 1:
                                mean_excl = (corr_mag.sum() - corr_mag[max_idx_local]) / (corr_mag.size - 1)
                            else:
                                mean_excl = 0.0

                            factor = peak_val / (mean_excl + 1e-12)

                            peak_thresh = NORM_PEAK_THRESH

                            print(
                                f"[CORR] peak_norm={peak_val:.6f} at corr_index={corr_index} "
                                f"(mean_excl={mean_excl:.6e}, factor={factor:.2f}, thresh={peak_thresh:.3f})"
                            )

                            end_lfm_idx = corr_index + LFM_LEN
                            guard_start_idx = end_lfm_idx + GUARD_SAMPLES
                            offset_samples = corr_filled - guard_start_idx

                            # Map buffer indices to an absolute sample number for easier verification.
                            # corr_buf[0] corresponds to absolute sample: sample_counter - corr_filled
                            abs_buf0 = sample_counter - corr_filled
                            abs_guard_start = abs_buf0 + guard_start_idx
                            abs_fsk_end = abs_guard_start + RECORD_SAMPLES

                            if (peak_val >= peak_thresh and
                                factor >= FACTOR_THRESH and
                                not recording_active and
                                offset_samples >= 0):
                                print(
                                    f"[SYNC_DEBUG] sample_counter={sample_counter}, corr_filled={corr_filled}, "
                                    f"corr_index={corr_index}, LFM_LEN={LFM_LEN}, GUARD_SAMPLES={GUARD_SAMPLES},"
                                    f" guard_start_idx={guard_start_idx}, offset={offset_samples},"
                                )
                                print(
                                    f"[SYNC_DEBUG] abs_guard_start={abs_guard_start}, abs_fsk_end={abs_fsk_end}, "
                                    f"expected_FSK_length={RECORD_SAMPLES}"
                                )
                                if offset_samples > 0:
                                    print(
                                        f"[SYNC] 'Now' is {offset_samples} samples AFTER "
                                        f"guard start (LFM end + {GUARD_SAMPLES} samples)."
                                    )
                                elif offset_samples == 0:
                                    print(
                                        "[SYNC] 'Now' is exactly at guard start (LFM end + 1s)."
                                    )

                                # store the absolute FSK start so saved blocks are aligned
                                pending_fsk_abs_start = int(abs_guard_start)

                                record_buffer = []
                                record_index = 0

                                if raw_buf is not None and raw_filled > 0:
                                    start_idx = max(0, min(guard_start_idx, raw_filled - 1))
                                    already_available = max(0, raw_filled - start_idx)
                                    pre_n = min(already_available, RECORD_SAMPLES)

                                    for i in range(pre_n):
                                        buf_idx = start_idx + i
                                        if buf_idx >= raw_filled:
                                            break
                                        sample_val = int(raw_buf[buf_idx])
                                        record_buffer.append((record_index, sample_val))
                                        record_index += 1

                                    record_samples_remaining = RECORD_SAMPLES - pre_n

                                    print(
                                        f"[SYNC] Pre-filled {pre_n} samples from history "
                                        f"starting at guard_start_idx={start_idx}."
                                    )

                                    if record_samples_remaining == 0:
                                                print(f"[REC] History already has full {RECORD_SAMPLES} samples "
                                                      f"for detection #{decode_index}.")
                                                # assemble numeric snapshot from record_buffer (values only)
                                                vals = np.array([v for (_, v) in record_buffer], dtype=np.int32)

                                                # enqueue write of aligned recorded block using pending_fsk_abs_start
                                                try:
                                                    abs_start = int(pending_fsk_abs_start) if pending_fsk_abs_start is not None else (sample_counter - len(vals))
                                                except Exception:
                                                    abs_start = sample_counter - len(vals)

                                                fname = os.path.join(DIAG_SAVE_DIR, f"det_{det_counter}.txt")
                                                det_counter += 1
                                                writer_queue.put((fname, vals, abs_start, FS))
                                                print(f"[WRITER] Enqueued recorded block write: {fname} (abs_start={abs_start}, samples={len(vals)})")

                                                if not DISABLE_DEMOD:
                                                    process_recorded_block(record_buffer, decode_index)
                                                else:
                                                    print("[DEMOD] Demodulation disabled; skipping processing of recorded block.")
                                                decode_index += 1
                                                recording_active = False
                                                record_buffer = []
                                                record_index = 0
                                    else:
                                        recording_active = True
                                        print(
                                            f"[REC] Detection #{decode_index} triggered. "
                                            f"Will capture remaining {record_samples_remaining} samples."
                                        )
                                else:
                                    recording_active = True
                                    record_samples_remaining = RECORD_SAMPLES
                                    record_buffer = []
                                    record_index = 0
                                    print(
                                        "[SYNC] No raw history; starting capture at 'now'.\n"
                                        f"[REC] Detection #{decode_index} triggered. "
                                        f"Will capture {record_samples_remaining} samples."
                                    )

                        samples_since_last_corr = 0

            # --- Spectrogram update ---
            if buf_filled >= NFFT:
                ax.clear()

                if buf_filled < plot_window_samples:
                    data_plot = plot_buf[:buf_filled]
                else:
                    data_plot = plot_buf

                data_plot = data_plot - np.mean(data_plot)

                Pxx, freqs, bins, im = ax.specgram(
                    data_plot,
                    NFFT=NFFT,
                    Fs=FS,
                    noverlap=NOVERLAP,
                    window=spec_window,
                    scale='dB',
                    mode='psd',
                    cmap='jet',
                    vmin=SPEC_VMIN_DB,
                    vmax=SPEC_VMAX_DB,
                )

                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Frequency [Hz]")
                ax.set_title(f"Real-time Spectrogram Hz)")
                ax.set_ylim(0, FS / 2)

                nonlocal_cb = cb
                if nonlocal_cb is None:
                    nonlocal_cb = fig.colorbar(im, ax=ax)
                    nonlocal_cb.set_label("Power/Frequency (dB/Hz)")
                    cb = nonlocal_cb
                else:
                    nonlocal_cb.update_normal(im)

                if decoded_text_artist is not None:
                    try:
                        decoded_text_artist.remove()
                    except Exception:
                        pass
                    decoded_text_artist = None
                if SHOW_DECODED_ON_PLOT and latest_decoded_text:
                    decoded_text_artist = ax.text(
                        0.02,
                        0.98,
                        latest_decoded_text,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=10,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
                    )

            plt.pause(0.001)

            # --- Periodic diagnostic snapshot ---
            if DIAG_PERIODIC_SAVE:
                try:
                    now_ts = time.time()
                    if now_ts - last_diag_save_ts >= DIAG_SAVE_INTERVAL_SEC:
                        # If configured to record full 30s files, require the full buffer
                        if RECORD_FULL_30SEC:
                            if full_filled < FULL_REC_SAMPLES:
                                print(f"[DIAG] Full 30s buffer not yet filled ({full_filled}/{FULL_REC_SAMPLES}), skipping save.")
                                last_diag_save_ts = now_ts
                                continue
                            # assemble chronological snapshot from circular buffer
                            if full_write_ptr == 0:
                                snapshot = full_buf.copy()
                            else:
                                snapshot = np.concatenate((full_buf[full_write_ptr:], full_buf[:full_write_ptr]))
                        else:
                            # prefer raw history if available, else use plot buffer
                            if raw_buf is not None and raw_filled > 0:
                                # save the currently held raw history (only filled portion)
                                snapshot = raw_buf[:raw_filled].astype(np.int32)
                            else:
                                snapshot = plot_buf.copy()

                        # sequential adc_N naming and enqueue write to background writer
                        fname = os.path.join(DIAG_SAVE_DIR, f"adc_{diag_counter}.txt")
                        diag_counter += 1
                        start_abs = sample_counter - len(snapshot) if len(snapshot) > 0 else sample_counter
                        # enqueue to writer (writer will write 2-column files: index value)
                        writer_queue.put((fname, snapshot, int(start_abs), FS))
                        print(f"[DIAG] Enqueued diagnostic snapshot: {fname} (samples={len(snapshot)})")
                        last_diag_save_ts = now_ts
                except Exception as e:
                    print(f"[DIAG] Unexpected error in periodic save: {e}")

        except KeyboardInterrupt:
            break

    print("\n[!] Stopping...")


# ============================================================
# MAIN ENTRY
# ============================================================

if __name__ == "__main__":
    # Make sure results file has header
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("timestamp,index,temperature,conductivity,pressure\n")

    try:
        lfm_normalized = generate_lfm_ref(FS)
        print(f"[INIT] Generated LFM reference in code, length = {len(lfm_normalized)}")
    except Exception as e:
        print(f"[ERROR] Could not generate LFM reference: {e}")
        lfm_normalized = None

    reader_thread = threading.Thread(target=serial_reader, daemon=True)
    reader_thread.start()

    try:
        main(lfm_normalized=lfm_normalized)
    finally:
        # signal threads to stop and flush writer queue
        running = False
        reader_thread.join()

        # wait for any pending writer tasks to complete, then stop writer
        try:
            print("[SHUTDOWN] Waiting for writer queue to drain...")
            writer_queue.join()
        except Exception:
            pass
        writer_running = False
        try:
            writer_thread.join(timeout=2.0)
        except Exception:
            pass

        print("[DONE]")
