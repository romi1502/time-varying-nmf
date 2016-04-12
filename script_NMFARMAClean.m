% script to test NMF with time-varying ARMA activations as described in "NMF 
% with time-frequency activations to model non stationary audio events" 
% to be published in IEEE Transactions on Audio Speech and Language Processing
%
% Copyright (C) 2010 Romain Hennequin

close all
clear all

%% parameters

% name of the file to get data from
fileName = 'Jew Harp';

% first sample of the sound
firstSample = 1; 

% last sample of the sound
lastSample = inf;

% decimation factor
decimateFactor = 4;

% size of fft in the spectrogram
Nfft = 1024;

% overlap factor (in [0,1[)
overlapFactor = 0.75;

% beta-divergence used
beta = 1;

% number of templates (atoms)
R = 1;

% order of MA filter
Q = 0;

% order of AR filter
P = 2;

% number of iterations
Niter = 50;



%% data preparation

% import
[x,fs] = wavread(fileName);

% transforming x in mono signal
x = toMono(x);

% selecting the samples to analyse :
x = x(firstSample:min(end,lastSample));
x = resample(x,1,decimateFactor,1000);
x = [zeros(Nfft/2,1); x];

% computation of spectrogram
sp = stft(x,Nfft,hanning(Nfft),Nfft*(1-overlapFactor));
V = abs(sp).^2;

% size of spectrogram
M = size(V,1);
N = size(V,2);


%% computation of the decomposition

[W, B, A, sig2] = NMFFARMAclean(V,R,Q,P,Niter,beta);
    

%% computation of values in order to display results

complexSinusT = zeros(M,Q+1);
for f=1:M
    complexSinusT(f,:) = exp(1i*2*pi*(f-1)/(2*M)*(0:Q));
end

complexSinusU = zeros(M,P+1);
for f=1:M
    complexSinusU(f,:) = exp(1i*2*pi*(f-1)/(2*M)*(0:P));
end


% computation of frequency response for each template and time
MAresponse = zeros(M,R,N);
ARresponse = zeros(M,R,N);
totalActivation = zeros(M,R,N);


Lambda = zeros(size(V));
for t=1:N
    for r=1:R
        ARresponse(:,r,t) = abs(complexSinusU*A(:,r,t)).^2;
        MAresponse(:,r,t) = abs(complexSinusT*B(:,r,t)).^2;
        totalActivation(:,r,t) = sig2(r,t).*MAresponse(:,r,t)./ARresponse(:,r,t);
        Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*MAresponse(:,r,t)./(ARresponse(:,r,t));
    end
end


%% spectrogram plot
scrsz = get(0,'ScreenSize');
figure('outerposition',scrsz)
subplot(211)
imagesc((0:N-1)*M*decimateFactor/(2*fs),(0:1/M:0.5)*fs/decimateFactor,db(V,'power'))
title('Original power spectrogram','fontSize',24)
axis xy;
xlabel('time (seconds)','fontsize',26);
ylabel('frequency (Hz)','fontsize',26);
dynamicData = max(caxis,-35);
caxis(dynamicData)
colorbar
colormap(flipud(gray(100)))

subplot(212)
imagesc((0:N-1)*M*decimateFactor/(2*fs),(0:1/M:0.5)*fs/decimateFactor,db(Lambda,'power'));
title('reconstructed power spectrogram','fontsize',24)
axis xy
xlabel('time (seconds)','fontsize',26);
ylabel('frequency (Hz)','fontsize',26);
caxis(dynamicData);
colormap(flipud(gray(100)))
colorbar
drawnow


%% T/F activation
scrsz = get(0,'ScreenSize');
figure('outerposition',scrsz);
for r = 1:min(R,4)
    subplot(min(R,4),1,r)
    imagesc((0:N-1)*M*decimateFactor/(2*fs),(0:1/(2*M):0.5)*fs/decimateFactor,db(reshape(totalActivation(:,r,:),M,N),'power'))
    title(['activation of template ' int2str(r)],'fontsize',24)
    xlabel('time (seconds)','fontsize',26)
    ylabel('frequency (Hz)','fontsize',26)
    axis xy
    dynamicData = max(caxis,-20);
    caxis(dynamicData)
    colormap(flipud(gray(100)))
    colorbar
end


%% frequency templates plot
figure
for r = 1:R
    subplot(R,1,r)
    hold on
    plot((0:1/(2*M-1):0.5)*fs/decimateFactor,db(W(:,r),'power'),'r')
    title(['template ' int2str(r)],'fontsize',24)
    xlabel('frequency (Hz)','fontsize',26)
    ylabel('magnitude (dB)','fontsize',26)
    ax = get(gcf,'children');
    set(ax(1),'fontsize',23)
end