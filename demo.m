function demo(varargin)
% DEMO  Real-time audio → STFT → (optional) neural net → live plots.
%
% This demo reads an incoming audio stream, computes a streaming STFT,
% (optionally) runs a neural network enhancement in a background worker,
% and plots both raw and processed spectrograms along with the waveform.
%
% Requirements:
%   - Audio Toolbox (audioDeviceReader, dsp.STFT)
%   - Parallel Computing Toolbox (parfeval) for async inference
%   - A ring buffer class with the following minimal API:
%       buf = RingBuffer(numRows, numCols)
%       buf.write(frame)                         % append along time axis
%       y = buf.plotYield()                      % returns most recent window
%       buf.framesTotal                          % total frames ever written
%       buf.framesProcessed                      % frames already processed
%       buf.processSamplesYield(nNew, nContext)  % read window for inference
%       buf.updateBufferFrameParametersPostProcessing(nJustProcessed)
%
% Notes:
%   - Replace RingBuffer(...) with your actual class (e.g., myBuffer).
%   - The included inferNeuralNet() stub runs in "TESTING" mode by default
%     and simply mimics latency. Set TESTING = 0 and provide an ONNX model
%     + normalization settings to enable real inference.

%% -------------------- Configuration --------------------

cfg = struct();
% Device & audio
cfg.useDevice         = true;     % set false to use your own reader function
cfg.fsWave            = 48e3;     % 48 kHz if using device; (16e3 also OK)
cfg.frameDuration     = 10e-3;    % 10 ms IO chunk size
cfg.numChannels       = 1;        % single-channel mic

% STFT
cfg.winDuration       = 32e-3;    % 32 ms
cfg.hopDuration       = 10e-3;    % 10 ms hop
cfg.nFFT              = [];       % [] → default = 2*winSamples (set below)
cfg.dbFloor           = -100;     % dB floor for visualization
cfg.dbCeil            = -25;      % dB ceiling for visualization
cfg.eps               = 1e-12;

% NN I/O framing (in STFT frames)
%   For 10 ms hop, inputTimeSteps=597 ≈ 5.97 s context
cfg.freqBins          = [];       % [] → auto from nFFT/2+1
cfg.inputTimeSteps    = 597;      % full context length for first call
cfg.inferStride       = 50;       % run inference every N frames after first
cfg.testingMode       = true;     % true = no model, mimic latency

% Display
cfg.displaySeconds    = 10;       % rolling window on plots
cfg.refreshRateHz     = 5;        % UI refresh rate (Hz)
cfg.waveLims          = [-1 1];   % waveform y-limits
cfg.colormapName      = 'parula'; % nicer than 'jet'

% Parallel
cfg.numWorkers        = 1;        % exactly one background worker

% Buffer class name (rename if your class is myBuffer)
cfg.BufferClass       = 'RingBuffer';  % e.g., 'myBuffer' if that's your class

% Parse overrides from varargin if desired (name-value)
if ~isempty(varargin)
    cfg = local_updateStruct(cfg, varargin{:});
end

%% -------------------- Derived parameters --------------------

fs       = cfg.fsWave;
winSamp  = round(cfg.winDuration * fs);
hopSamp  = round(cfg.hopDuration * fs);
frameLen = max(hopSamp, 1);

if isempty(cfg.nFFT)
    cfg.nFFT = 2 * winSamp;
end
freqBins = cfg.nFFT/2 + 1;
if isempty(cfg.freqBins)
    cfg.freqBins = freqBins;
end

fsSTFT = 1 / cfg.hopDuration;  % frames/sec for STFT

%% -------------------- Setup: audio / parallel / STFT --------------------

% Audio reader
if cfg.useDevice
    deviceReader = audioDeviceReader( ...
        'SampleRate',      fs, ...
        'SamplesPerFrame', frameLen, ...
        'NumChannels',     cfg.numChannels);
    cleanupAudio = onCleanup(@() release(deviceReader));
else
    % Provide your own function handle that returns [x, overrun]
    % Example: deviceReader = @myAudioDevice;  % returns 1xT and overrun count
    error('cfg.useDevice=false requires a user-provided audio reader.');
end

% Parallel pool: exactly one worker
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= cfg.numWorkers || ~strcmp(pool.Cluster.Type, "local")
    delete(gcp('nocreate'));
    pool = parpool("local", cfg.numWorkers); %#ok<NASGU>
end

% Pre-warm model in background (even in testing mode to unify flow)
fWarm = parfeval(@inferNeuralNet, 0, 'warmup', [], cfg.testingMode);
fetchOutputs(fWarm);

% Streaming STFT object
stftObj = dsp.STFT( ...
    'Window',         hamming(winSamp, 'periodic'), ...
    'OverlapLength',  winSamp - hopSamp, ...
    'FFTLength',      cfg.nFFT, ...
    'FrequencyRange', 'onesided');

%% -------------------- Display buffers & axes --------------------

% Ring buffers hold a rolling window of ~displaySeconds
TdispFrames = round(cfg.displaySeconds * fsSTFT);
TdispSamps  = round(cfg.displaySeconds * fs);

bufWave     = feval(cfg.BufferClass, 1,                TdispSamps);
bufStftRaw  = feval(cfg.BufferClass, cfg.freqBins,     TdispFrames);
bufStftProc = feval(cfg.BufferClass, cfg.freqBins,     TdispFrames);

% Time axes (left-aligned at -displaySeconds)
tWave   = local_timeAxis(TdispSamps, fs,      -cfg.displaySeconds);
tStft   = local_timeAxis(TdispFrames, fsSTFT, -cfg.displaySeconds);

% Figure & axes
hFig  = figure('Position', [0 0 1200 950], 'Color', 'w', 'Name', 'Real-Time STFT Demo');
cleanupFig = onCleanup(@() local_safeClose(hFig)); %#ok<NASGU>
tlo   = tiledlayout(hFig, 7, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Waveform: spans first row (except last tile)
axWave = nexttile(tlo, [1 3]); %#ok<NASGU>
hWave  = plot(tWave, zeros(size(tWave))); grid on;
ylim(cfg.waveLims); xlim([-cfg.displaySeconds 0]);
title('Waveform'); xlabel('Time (s)'); ylabel('Amplitude');

% Raw STFT: big block
axRaw  = nexttile(tlo, [2 3]); %#ok<NASGU>
hRaw   = imagesc(tStft, [], -80*ones(cfg.freqBins, TdispFrames));
axis xy; xlim([-cfg.displaySeconds 0]);
clim([cfg.dbFloor cfg.dbCeil]); colormap(cfg.colormapName);
title('Raw STFT (dB)'); xlabel('Time (s)'); ylabel('Frequency bin');

% Processed STFT: big block
axProc = nexttile(tlo, [2 3]); %#ok<NASGU>
hProc  = imagesc(tStft, [], -80*ones(cfg.freqBins, TdispFrames));
axis xy; xlim([-cfg.displaySeconds 0]);
clim([cfg.dbFloor cfg.dbCeil]); colormap(cfg.colormapName);
title('Processed STFT (dB)'); xlabel('Time (s)'); ylabel('Frequency bin');

%% -------------------- Streaming loop --------------------

lastUpdate   = tic;
dtRefresh    = 1 / cfg.refreshRateHz;

firstInfer   = false;
pendingFut   = [];
totalOverrun = 0;
frameCount   = 0;

while ishghandle(hFig)
    % -------- Read audio --------
    if cfg.useDevice
        x = deviceReader();           % [Nsamp x Nch]
        x = x(:, 1).';                % keep 1 channel, row vec 1xT
        samplesOverrun = 0;           % audioDeviceReader tracks dropouts internally
    else
        [x, samplesOverrun] = deviceReader(); % custom reader must match this API
        x = x(:).';                               % ensure row
    end

    if samplesOverrun > 0
        totalOverrun = totalOverrun + samplesOverrun;
    end
    frameCount = frameCount + 1;

    % -------- STFT (magnitude→dB) --------
    X = stftObj(x.');              % [F x K], complex
    if ~isempty(X)
        Xdb = 20*log10(abs(X) + cfg.eps);
    else
        Xdb = [];
    end

    % -------- Write to buffers --------
    bufWave.write(x);
    if ~isempty(Xdb)
        bufStftRaw.write(Xdb);
    end

    % -------- Neural net inference (async) --------
    haveEnoughForFirst = (bufStftRaw.framesTotal > cfg.inputTimeSteps);
    haveStrideForNext  = (bufStftRaw.framesTotal - bufStftRaw.framesProcessed) > cfg.inferStride;

    if ~firstInfer && haveEnoughForFirst
        % First full-window inference
        if isempty(pendingFut)
            block = bufStftRaw.processSamplesYield(cfg.inputTimeSteps, 0);
            pendingFut = parfeval(@inferNeuralNet, 1, 'infer', block, cfg.testingMode);
        elseif strcmp(pendingFut.State, 'finished')
            Xest = fetchOutputs(pendingFut);
            bufStftProc.write(Xest);
            bufStftRaw.updateBufferFrameParametersPostProcessing(cfg.inputTimeSteps);
            firstInfer = true;
            pendingFut = [];
        end

    elseif firstInfer && haveStrideForNext
        % Subsequent sliding-window inference
        if isempty(pendingFut)
            block = bufStftRaw.processSamplesYield(cfg.inferStride, cfg.inputTimeSteps - cfg.inferStride);
            pendingFut = parfeval(@inferNeuralNet, 1, 'infer', block, cfg.testingMode);
        elseif strcmp(pendingFut.State, 'finished')
            Xest = fetchOutputs(pendingFut);
            % Only append the newly inferred tail (stride frames)
            bufStftProc.write(Xest(:, end-cfg.inferStride+1:end));
            bufStftRaw.updateBufferFrameParametersPostProcessing(cfg.inferStride);
            pendingFut = [];
        end
    end

    % -------- UI refresh --------
    if toc(lastUpdate) >= dtRefresh && bufWave.framesTotal > fs*5
        set(hWave, 'YData', bufWave.plotYield());
        set(hRaw,  'CData', bufStftRaw.plotYield());
        set(hProc, 'CData', bufStftProc.plotYield());
        drawnow limitrate;
        lastUpdate = tic;
    end
end

% Cleanup messages
fprintf('[demo] Total overruns (samples): %d (%.2f s)\n', totalOverrun, totalOverrun/fs);

end % demo()


%% ======================== Helper functions ========================

function x_clean = inferNeuralNet(mode, x_noisy, TESTING)
% Background worker entry point for warmup/inference.
% Inputs:
%   mode     : 'warmup' | 'infer'
%   x_noisy  : [F x T] spectrogram (dB) for 'infer'; [] for 'warmup'
%   TESTING  : logical; if true, simulate latency & pass-through
%
% Output:
%   x_clean  : processed spectrogram (dB), same size as x_noisy

% Persistent handles for real inference (set TESTING=false and fill values)
persistent net clip_db_min clip_db_max mu_db sigma_db initialized

if nargin < 3, TESTING = true; end
x_clean = [];

switch mode
    case 'warmup'
        if TESTING
            pause(0.05); % fast "warm"
            initialized = true;
            return;
        end

        % ------- REAL MODEL PATHS / PARAMS (EDIT ME) -------
        modelPath   = 'models/best_model.onnx';   % <- put your model in repo
        clip_db_min = -100;                       % clamp before norm
        clip_db_max = 0;
        mu_db       = -62.14630865105698;         % dataset mean (example)
        sigma_db    = 11.96422421642002;          % dataset std  (example)

        % Import & initialize
        net = importNetworkFromONNX(modelPath, ...
            InputDataFormats='BCSS', OutputDataFormats='BCSS');
        tmp = rand(1,1,513,597,'single');           % [B,C,F,T]
        net = initialize(net, dlarray(tmp,'BCSS'));
        initialized = true;

    case 'infer'
        if TESTING
            % Mimic ~200 ms compute time; pass-through
            pause(0.20);
            x_clean = x_noisy;
            return;
        end

        assert(~isempty(initialized), 'inferNeuralNet: call warmup first.');

        % Preprocess (clamp + normalize)
        x = x_noisy;
        x(x > clip_db_max) = clip_db_max;
        x(x < clip_db_min) = clip_db_min;
        x = (x - mu_db) / max(sigma_db, 1e-12);  %#ok<NASGU>

        % Ensure [B,C,F,T]
        [F,T] = size(x);
        x = reshape(x, [1 1 F T]);
        dlX = dlarray(single(x), 'BCSS');

        % Predict
        dlY = predict(net, dlX);
        y   = gather(extractdata(dlY));   % [1,1,F,T]
        y   = squeeze(y);                 % [F,T]

        % De-normalize
        x_clean = y * sigma_db + mu_db;

    otherwise
        error('inferNeuralNet: unknown mode "%s"', mode);
end
end


function t = local_timeAxis(N, fs, tStart)
% Create a time axis of length N with sample rate fs starting at tStart (s).
t = tStart + (0:N-1) / fs;
end


function local_safeClose(hFig)
if ishghandle(hFig)
    try, close(hFig); catch, end
end
end


function s = local_updateStruct(s, varargin)
% Update struct fields from name-value pairs; ignore unknown fields.
if mod(numel(varargin),2) ~= 0
    error('Name-value arguments must come in pairs.');
end
for k = 1:2:numel(varargin)
    name  = varargin{k};
    value = varargin{k+1};
    if isfield(s, name)
        s.(name) = value;
    else
        warning('Unknown config field "%s" ignored.', name);
    end
end
end
