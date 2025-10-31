# Real-Time Neural Network Inference on Streaming Time-Series Data

This repository demonstrates **real-time neural network inference** on continuous time-series data (e.g., microphone input) using MATLAB. It processes an **incoming audio stream**, computes a **Short-Time Fourier Transform (STFT)** in real time, and performs **neural network enhancement/denoising** before visualizing results live.

---

## 🚀 Key Features

- **Real-Time Processing:** Streams data continuously from a live audio device.
- **STFT Visualization:** Displays the input waveform, raw STFT, and processed STFT in real time.
- **Neural Network Integration:** Supports asynchronous model inference using MATLAB’s `parfeval`, allowing non-blocking neural network computation (via ONNX models).
- **Custom Ring Buffer:** Includes a robust multi-channel circular buffer (`myBuffer.m`) for managing streaming data with overlapping context windows.
- **Live Display:** Visualizes both input and processed spectrograms, updating several times per second.

---

## 🧠 Core Files

| File | Description |
|------|-------------|
| **`demo.m`** | Main script that streams audio, computes STFT, runs neural net inference asynchronously, and plots results live. |
| **`myBuffer.m`** | Custom multi-channel circular buffer class for managing streaming input and processed data. |

---

## 🧩 System Overview

```text
 ┌────────────┐     ┌────────────┐     ┌──────────────┐     ┌───────────────┐
 │ Audio Mic  │ →→→ │ STFT Block │ →→→ │  Neural Net  │ →→→ │ Real-Time GUI │
 │(Live Input)│     │  (Sliding) │     │ (ONNX model) │     │  Wave + STFT  │
 └────────────┘     └────────────┘     └──────────────┘     └───────────────┘
```

Internally, the demo maintains three ring buffers:
- `bufferWaveform` — raw time-domain samples
- `bufferSTFTRaw` — input spectrogram
- `bufferSTFTProcessed` — neural net–enhanced spectrogram

Each buffer continuously updates and overwrites old data, keeping the visualization smooth and low latency.

---

## 🧰 Requirements

- MATLAB R2023a or later
- Toolboxes:
  - Audio Toolbox
  - Parallel Computing Toolbox
  - DSP System Toolbox
  - (Optional) Deep Learning Toolbox for loading ONNX models

---

## ⚙️ Usage

Clone this repository:

```bash
git clone https://github.com/vishalchoudhari11/Real-Time-Neural-Net-Inference-Demo.git
cd real-time-audio-inference
```

Open MATLAB and add this folder to the path:

```matlab
addpath(genpath(pwd));
```

Run the demo:

```matlab
demo;
```

To use your own neural network:
1. Export your trained model to ONNX.
2. Place it under the `models/` folder.
3. Update the path in `inferNeuralNet()` inside `demo.m`.

---

## 🧩 The Buffer Class (`myBuffer.m`)

The `myBuffer` class provides an efficient circular buffer interface for managing multi-channel streaming data. It supports:

- Continuous writes and overwrites
- Chronological readouts for plotting
- Windowed reads for neural net inference
- Safe update of frame counters post-processing

Example:

```matlab
buf = myBuffer(1, 16000);       % 1 channel, 1-second buffer at 16 kHz
buf.write(randn(1, 4000));      % write samples
data = buf.plotYield();         % get chronological view
imagesc(data);
```

---

## 🧠 Real-Time Neural Network Inference

The demo can perform asynchronous neural net inference:

- The STFT stream is passed to a background worker using `parfeval`.
- Inference can run using any ONNX-compatible model (e.g., speech enhancement, denoising, separation).
- The main thread continues streaming and updating the display without waiting for inference to finish.

```matlab
% Inside demo.m
pending = parfeval(@inferNeuralNet, 1, 'infer', bufferSTFTRaw.processSamplesYield(...));
```

To adapt this for other streaming time-series data (e.g., physiological sensors, radar, EEG):

- Replace the audio input section with your data source.
- Adjust STFT parameters or preprocessing to match your domain.

---

## 📊 Real-Time Visualization

The GUI displays:

- Waveform plot: Time-domain view of the recent buffer window.
- Raw STFT spectrogram: Frequency–time magnitude before processing.
- Processed STFT spectrogram: Output from neural net (enhanced or denoised).

---

## 🧩 Typical Use Cases

- Real-time speech enhancement / denoising
- EEG / EMG / physiological time-series processing with sliding neural inference

---

## 📁 Recommended Folder Structure

```
real-time-audio-inference/
├── demo.m
├── myBuffer.m
├── models/
│   └── best_model.onnx
└── README.md
```

---

## ⭐ Contributions

Pull requests and improvements are welcome! If you extend this framework (e.g., add new neural networks or visualization modes), please open a PR or issue.

