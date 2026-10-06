# MI receiver

Standalone serial receiver for LFM chirp detection and BFSK sensor telemetry.
`integrated_dsp.py` reads ADC samples, downsamples them, displays a live
spectrogram, detects the chirp, and demodulates and Viterbi-decodes captured data.
It generates the LFM reference internally; no `LFM_REF.txt` or other project
scripts are required.

## Setup and run (Windows PowerShell)

Use Python 3.11 and run these commands from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe integrated_dsp.py
```

Connect the ADC/receiver hardware and edit the configuration constants at the
top of `integrated_dsp.py` before running. A graphical desktop is required for
the live Matplotlib display. Stop with Ctrl+C in the terminal.

## Current configuration

| Setting | Default |
| --- | --- |
| Serial port | `COM9` |
| Baud rate | `921600` |
| Input format | Little-endian 16-bit words, interpreted as signed samples |
| ADC sample rate | 16000 Hz |
| Processing sample rate | 8000 Hz |
| LFM carrier / bandwidth / duration | 950 Hz / 100 Hz / 1.024 s |
| BFSK tones / symbol duration | 900 and 1000 Hz / 0.02 s |
| Guard after chirp | 1 s |
| Capture duration | 2.76 s |

The serial format, sample rate, chirp, tones, and timing must match the
transmitter and ADC firmware. Detection thresholds are also configured in the
script. The default demodulator uses time-domain correlations; the optional
spectrogram demodulator is selected by `USE_SPECTROGRAM_DEMOD`.

## Generated files

Paths are relative to the working directory, so launch from this folder.

- `decoded_results.txt`: CSV-formatted rows with
  `timestamp,index,temperature,conductivity,pressure`.
- `diagnostics/det_*.txt`: detection captures.
- `diagnostics/adc_*.txt`: optional periodic diagnostic/full recordings,
  controlled by the recording flags.

Diagnostic recordings contain two columns: absolute sample index and sample
value. Recordings, result logs, Python caches, and virtual environments are
excluded by `.gitignore`.

## Current behavior and validation limits

This is the existing receiver implementation, with the Tukey window imported
from the supported `scipy.signal.windows` namespace:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html

Conductivity and depth are currently set to zero rather than decoded from their
payload fields. Temperature values outside the strict 10-50 degree range are
replaced with 20.2. A missing sync word prints a warning but does not reject the
frame. Keep `USE_HANNING_DEMOD=True` for the default correlator: its existing
implementation uses the window unconditionally.

Hardware reception, GUI operation, and decoding accuracy must be checked with
the actual transmitter/ADC setup. Dependency/import checks alone do not validate
those behaviors.

Validated with Python 3.11 and the pinned dependencies: module imports, LFM
reference length/normalization, 16-to-8 kHz downsampling, and synthetic BFSK
tone decisions. No serial hardware or live GUI session was exercised.
