classdef myBuffer < handle
%MYBUFFER  Simple multi-channel circular (ring) buffer for streaming data.
%
%   Time flows along the 2nd dimension (columns).  New samples overwrite
%   the oldest when the buffer wraps around.
%
%   Example:
%       buf = myBuffer(1, 16000);
%       buf.write(randn(1,4000));
%       buf.write(randn(1,4000));
%       imagesc(buf.plotYield());
%
%   Public properties:
%       nChannels        - number of channels (rows)
%       Samples          - buffer length (columns)
%       buffer            [nChannels x Samples]
%       writePTR          next column to write (1..Samples)
%       isFull            true once buffer wraps
%       framesTotal       total frames written since creation
%       framesProcessed   total frames already processed
%       lastProcessedPTR  column index of the last processed frame (0=none)
%
%   Key methods:
%       write(frames)                       % append new data
%       plotYield() -> data                 % get oldest→newest view
%       processSamplesYield(new, old)       % window for inference
%       updateBufferFrameParametersPostProcessing(nNew)
%       newFramesInBuffer() -> k
%
%   Notes:
%       • Columns represent time.
%       • latest frame index = writePTR-1 (wrapped)
%       • Frames that arrive beyond capacity overwrite oldest ones.

    properties
        nChannels
        Samples
        buffer

        writePTR
        isFull

        framesTotal
        framesProcessed
        lastProcessedPTR
    end

    methods
        function self = myBuffer(nChannels, Samples)
            % Constructor
            arguments
                nChannels (1,1) double {mustBePositive, mustBeInteger}
                Samples   (1,1) double {mustBePositive, mustBeInteger}
            end
            self.nChannels = nChannels;
            self.Samples   = Samples;
            self.buffer    = zeros(nChannels, Samples);
            self.writePTR  = 1;
            self.isFull    = false;
            self.framesTotal       = 0;
            self.framesProcessed   = 0;
            self.lastProcessedPTR  = 0;
        end

        function write(self, frames)
            %WRITE  Append samples to the ring buffer.
            %   frames must be [nChannels x N].
            [C, L] = size(frames);
            assert(C == self.nChannels, ...
                'Input must have %d rows (channels).', self.nChannels);

            % If more than capacity, keep most recent window
            if L >= self.Samples
                frames = frames(:, end-self.Samples+1:end);
                L = self.Samples;
                warning('myBuffer:Overflow', ...
                    'Input exceeds capacity. Oldest samples discarded.');
            end

            startPTR  = self.writePTR;
            firstPart = min(L, self.Samples - self.writePTR + 1);

            % Write contiguous block
            if firstPart > 0
                self.buffer(:, self.writePTR:self.writePTR+firstPart-1) = ...
                    frames(:, 1:firstPart);
            end

            % Wrap remainder if needed
            remainder = L - firstPart;
            if remainder > 0
                self.buffer(:, 1:remainder) = frames(:, firstPart+1:end);
            end

            % Advance write pointer & bookkeeping
            self.writePTR = self.wrapIndex_(self.writePTR + L);
            self.framesTotal = self.framesTotal + L;

            % Mark full once buffer wraps or fills completely
            if ~self.isFull
                wrapped = (startPTR + L - 1) >= self.Samples;
                self.isFull = wrapped || (L == self.Samples);
            end
        end

        function data = plotYield(self)
            %PLOTYIELD  Return chronological data [nChannels x Samples].
            %   Oldest→newest. Left-pads with NaNs if buffer not yet full.
            if self.isEmpty_()
                data = nan(self.nChannels, self.Samples);
                return;
            end

            if self.isFull
                % Oldest at writePTR
                if self.writePTR == 1
                    order = 1:self.Samples;
                else
                    order = [self.writePTR:self.Samples, 1:self.writePTR-1];
                end
                data = self.buffer(:, order);
            else
                filled = self.writePTR - 1;
                padLen = self.Samples - filled;
                if filled <= 0
                    data = nan(self.nChannels, self.Samples);
                else
                    data = [nan(self.nChannels, padLen), ...
                            self.buffer(:, 1:filled)];
                end
            end
        end

        function data = processSamplesYield(self, newFrames, oldFrames)
            %PROCESSSAMPLESYIELD  Return a window for model inference.
            %
            %   data = processSamplesYield(new, old)
            %   Combines:
            %       • oldFrames  ending at lastProcessedPTR
            %       • newFrames  immediately after it
            %   Returns chronological [nChannels x (old+new)] block.
            %
            %   Call updateBufferFrameParametersPostProcessing(newFrames)
            %   after the inference completes.

            if self.isEmpty_()
                data = [];
                return;
            end

            if self.framesProcessed == 0
                assert(oldFrames == 0, ...
                    'First call must have oldFrames = 0.');
            end

            anchor = self.lastProcessedPTR;
            oldIdx = self.backwardSpan_(anchor, oldFrames);
            newIdx = self.forwardSpan_(anchor, newFrames);
            data   = [self.buffer(:, oldIdx), self.buffer(:, newIdx)];
        end

        function updateBufferFrameParametersPostProcessing(self, newFrames)
            %UPDATEBUFFERFRAMEPARAMETERSPROCESSING  Advance processing pointers.
            self.framesProcessed   = self.framesProcessed + newFrames;
            self.lastProcessedPTR  = self.wrapIndex_(self.framesProcessed);
        end

        function k = newFramesInBuffer(self)
            %NEWFRAMESINBUFFER  Number of unprocessed frames available.
            k = self.framesTotal - self.framesProcessed;
            if k > self.Samples
                warning('myBuffer:Overwrite', ...
                    'Frames were overwritten before processing.');
            end
        end
    end

    %% ================= Private helpers =================
    methods (Access = private)
        function tf = isEmpty_(self)
            tf = (~self.isFull) && (self.writePTR == 1);
        end

        function idx = wrapIndex_(self, k)
            idx = mod(k - 1, self.Samples) + 1;
        end

        function idx = backwardSpan_(self, anchor, L)
            if L <= 0, idx = zeros(1,0); return; end
            startIdx = self.wrapIndex_(anchor - L + 1);
            if startIdx <= anchor
                idx = startIdx:anchor;
            else
                idx = [startIdx:self.Samples, 1:anchor];
            end
        end

        function idx = forwardSpan_(self, anchor, L)
            if L <= 0, idx = zeros(1,0); return; end
            first = self.wrapIndex_(anchor + 1);
            last  = self.wrapIndex_(anchor + L);
            if first <= last
                idx = first:last;
            else
                idx = [first:self.Samples, 1:last];
            end
        end
    end
end
