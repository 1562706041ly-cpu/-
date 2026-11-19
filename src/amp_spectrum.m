function [amp, f] = amp_spectrum(y, fs)
%AMP_SPECTRUM  单边幅值谱
y = y(:).' - mean(y);
L = numel(y);
nfft = 2^nextpow2(L);
Y = fft(y, nfft);
amp = 2 * abs(Y(1:nfft/2)) / L;
f = fs * (0:nfft/2-1) / nfft;
end
