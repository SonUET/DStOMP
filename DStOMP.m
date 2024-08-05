clear; close all; clc;
%% This code is based on https://github.com/henkwymeersch/5GPositioning
%% System Params
L       = 2;                    % number of paths (This code only works with 2 paths)
Rs      = 100e6;                % total BW in Hz
N       = 5;                    % number of subcarriers
Nt      = 16;                   % number of TX antennas
Nr      = Nt;                   % number of RX antennas
Nb      = Nt*2;                 % number of beams in dictionary
G       = 10;                   % number of beams sent (randomly beams)
c       = 3e8;                  % speed of light m/s
Ts      = 1/Rs;                 % sampling period in us
posTx   = [0 0]';               % TX is assumed to be in [0, 0]
posRx   = [4 1]';               % RX (user) position
SP      = [2, 2];               % scatter point position
alpha   = 0.2;                  % user orientation
h       = 10*ones(1,L);         % channel gain
sigma   = 0.1;                  % noise std
%% Compute Channel Parameters for L paths
TOA = zeros(1, L); AOD = zeros(1, L); AOA = zeros(1, L);
TOA(1) = norm(posRx)/c;                                                     % LOS TOA
AOD(1) = atan2(posRx(2) - posTx(2), posRx(1) - posTx(1));                   % LOS AOD
AOA(1) = atan2(posTx(2) - posRx(2), posTx(1) - posRx(1)) - alpha;           % LOS AOA
for p = 1:L-1
    TOA(p+1) = (norm(SP(p,:)) + norm(posRx - SP(p,:)'))/c;                  % NLOS TOA
    AOD(p+1) = atan2(SP(p,2), SP(p,1));                                     % NLOS AOD
    AOA(p+1) = atan2(SP(p,2) - posRx(2), SP(p,1) - posRx(1)) - alpha;       % NLOS AOD
end
%% Generate dictionary
Ut = zeros(Nt,Nb);
Ur = zeros(Nr,Nb);
aa = -Nb/2:Nb/2-1;
aa = 2*aa/Nb;
for m = 1:Nb
    Ut(:,m) = getResponse(Nt,aa(m))*sqrt(Nt);
    Ur(:,m) = getResponse(Nr,aa(m))*sqrt(Nr);
end
Ut = Ut/sqrt(Nb);
Ur = Ur/sqrt(Nb);
%% Generate channel
H = zeros(Nr,Nt,N); A_rx = zeros(Nr,L); A_tx = zeros(Nt,L); Gamma = zeros(L, L, N);
Hb = zeros(Nb, Nb, N);
for n = 1:N
    for p = 1:L
        A_rx(:,p) = getResponse(Nr,sin(AOA(p)))*sqrt(Nr);
        A_tx(:,p) = getResponse(Nt,sin(AOD(p)))*sqrt(Nt);
        Gamma(p,p,n) = h(p)*exp(-1j*2*pi*TOA(p)*(n-1)/(N*Ts));
        H(:,:,n) = H(:,:,n) + A_rx(:,p)*Gamma(p,p,n)*A_tx(:,p)';
    end
    Hb(:,:,n) = Ur'*H(:,:,n)*Ut;
end
%% The beamspace channel
Hb = zeros(Nb, Nb, N);
for n = 1:N
    Hb(:,:,n) = Ur'*H(:,:,n)*Ut;
end
%% Generate the observation and beamformers
y = zeros(Nr,G,N); signal = zeros(Nr,G,N); noise = zeros(Nr,G,N); F = zeros(Nt,G,N);
for g = 1:G
    for n = 1:N
        F(:,g,n) = exp(1j*rand(Nt,1)*2*pi); % random beamformers (note: we don't add data symbols, they are part of F)
    end
end
for g = 1:G
    for n = 1:N
        signal(:,g,n) = H(:,:,n)*F(:,g,n);
        noise(:,g,n) = sigma/sqrt(2)*(randn(Nr,1) + 1i*randn(Nr,1));       % noise
        y(:,g,n) = signal(:,g,n) + noise(:,g,n);
    end
end
%% Vectorize and generation of the basis
yb = zeros(Nr*G,N);
Omega = zeros(Nr*G,Nb*Nb,N);
for n = 1:N
    yb(:,n) = reshape(y(:,:,n), Nr*G,1);
    Omega(:,:,n) = kron((Ut'*F(:,:,n)).',Ur);
end
y_vec = reshape(yb, [Nr*G*N 1]);
%% Distributed StOMP
r = yb;
iter = 1; maxIters = 10;
arg = zeros(size(Omega, 2), size(Omega, 3));
normy = norm(yb, 'fro');
I_s = []; h_hat = []; done = 0;
% Normalize columns of the sensing matrix
NOmega = zeros(size(Omega));
for n=1:N
    for i = 1:length(Omega)
        NOmega(:,i, n) = Omega(:,i, n)/norm(Omega(:,i, n));
    end
end
c_k = zeros(1, length(Omega));
while(~done)
    % Detect support set
    for n=1:N
        for i=1:size(NOmega, 2)
            arg(i, n) = abs((NOmega(:, i, n)'*r(:, n)));
        end
    end
    for i=1:length(arg)
        c_k(i) = sum(arg(i, :));
    end
    % Thresholding
    sigma_s = norm(r, 'fro')/sqrt(length(yb)*N);
    t_s = 4.2*N; 
    thr = sigma_s*t_s;
    I = find(abs(c_k) > thr);
    % Update the support set
    J = union(I_s, I);
    if (length(J) == length(I_s))
        done = 1;
    else
        phi_S = [];
        I_s = J;
     % Construct sensing matrix
        for n=1:N
            phi_S = blkdiag(phi_S, Omega(:, I_s, n));
        end
     % Project signal on subspace spanned by the support set
        h_hat = pinv((phi_S'*phi_S))*(phi_S'*y_vec);
     % Update residual
        r = r(:) - phi_S*h_hat;
        r = reshape(r, Nr*G, N);
    end
    iter = iter+1;
    % Check stopping criteria
    if (iter > maxIters)
        done = 1;
    end
    h_est = zeros(length(I_s),1);
    for i=1:length(I_s)
        for n=1:N
            h_est(i) = h_est(i) + abs(h_hat((n-1)*length(I_s)+i));
        end
    end
end

Hb_est = zeros(Nb*Nb, 1);
Hb_est(I_s) = h_est;
H_est = reshape(Hb_est, Nb, Nb);

subplot(121)
mesh(abs(Hb(:,:,1)))
xlabel('AOD index'); ylabel('AOA index'); zlabel('Magnitude')
title('Noise-free beamspace channel')
axis square
subplot(122)
mesh(abs(H_est)/(Nt));
xlabel('AOD index'); ylabel('AOA index'); zlabel('Magnitude')
axis square
title('Estimated channel using DStOMP algorithm only')
