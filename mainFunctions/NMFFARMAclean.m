function [W, B, A, sig2] = NMFFARMAclean(V,R,Q,P,Niter,beta,initialV)
% [W, B, A, sig2] = NMFFARMAclean(V,R,Q,P,Niter,beta,initialV)
%    NMFFARMA : NMF with ARMA filter activation as described in "NMF with time-frequency activations to model non stationary audio events" 
%    to be published in IEEE Transactions on Audio Speech and Language Processing
%Input :
%   - V : power spectrogram to factorize (a MxN matrix)
%   - R : number of templates
%   - Q : order of MA
%   - P : order of AR
%   - Niter : number of iterations
%   - beta (optional): beta used for beta-divergence (default : beta = 0,
%   IS divergence)
%   - initialV (optional) : initial values of W, B, A and sig2 (a struct
%   with
%   fields W, B, A and sig2)
%Output :
%   - W : frequency templates (MxR array)
%   - B : MA coefficients of filter activation for each template ((Q+1)xRxN array)
%   - A : AR coefficients of filter activation for each template ((P+1)xRxN array)
%   - sig2 : global gain of the ARMA filter (RxN array)
%
% Copyright (C) 2010 Romain Hennequin



% filters normalization type (true : power, false : first coefficient)
powerFiltersNormalization = false;

% size of input spectrogram
M = size(V,1);
N = size(V,2);

sig2Threshold = 10e-8;

% initialization
if nargin == 7
    B = initialV.B;
    A = initialV.A;
    W = initialV.W;
    sig2 = initialV.sig2;
else
    B = zeros(Q+1,R,N);
    B(1,:,:) = 1;
    A = zeros(P+1,R,N);
    A(1,:,:) = 1;
    sig2 = abs(randn(R,N));
    W = abs(randn(M,R));

    if nargin == 5
        beta = 0;
    end
end

% local variable
Lambda = zeros(M,N);

R1c = zeros(Q+1,1);
R2c = zeros(Q+1,1);

% toeplitz allocation variables
cidR = (0:Q)';
ridR = Q+1:-1:1;
indexToeplitzQ = cidR(:,ones(Q+1,1)) + ridR(ones(Q+1,1),:);

R1cr = [R1c(Q+1:-1:2) ; R1c];
R1 = R1cr(indexToeplitzQ);
R2cr = [R2c(Q+1:-1:2) ; R2c];
R2 = R2cr(indexToeplitzQ);

S1c = zeros(P+1,1);
S2c = zeros(P+1,1);

% toeplitz allocation variables
cidS = (0:P)';
ridS = P+1:-1:1;
indexToeplitzP = cidS(:,ones(P+1,1)) + ridS(ones(P+1,1),:);

S1cr = [S1c(P+1:-1:2) ; S1c];
S1 = S1cr(indexToeplitzP);
S2cr = [S2c(P+1:-1:2) ; S2c];
S2 = S2cr(indexToeplitzP);




Tcolumn = zeros(Q+1,M);
for f = 1:M
    Tcolumn(:,f) = cos(2*pi*(f-1)/(2*(M-1))*(0:Q))';
end

Ucolumn = zeros(P+1,M);
for f = 1:M
    Ucolumn(:,f) = cos(2*pi*(f-1)/(2*(M-1))*(0:P))';
end

complexSinusT = zeros(M,Q+1);
for f=1:M
    complexSinusT(f,:) = exp(1i*2*pi*(f-1)/(2*(M-1))*(0:Q));
end

complexSinusU = zeros(M,P+1);
for f=1:M
    complexSinusU(f,:) = exp(1i*2*pi*(f-1)/(2*(M-1))*(0:P));
end

MAresponse = ones(M,R,N);
ARresponse = ones(M,R,N);

% filter normalization
if nargin<7
    for r0=1:R
        for t0=1:N
            if powerFiltersNormalization
                GB = sqrt(sum(MAresponse(:,r0,t0)));
                GA = 1/sqrt(sum(1./ARresponse(:,r0,t0)));
            else
                GB = B(1,r0,t0);
                GA = A(1,r0,t0);
            end

            A(:,r0,t0) = A(:,r0,t0)/GA;
            ARresponse(:,r0,t0) = ARresponse(:,r0,t0)/abs(GA).^2;
            B(:,r0,t0) = B(:,r0,t0)/GB;
            MAresponse(:,r0,t0) =  MAresponse(:,r0,t0)/abs(GB).^2;
        end
    end
end



% computation of Lambda (estimate of V) and of filters repsonse
for t=1:N
    for r = 1:R
        ARloc = complexSinusU*A(:,:,t);            
        ARresponse(:,:,t) = real(ARloc.*conj(ARloc));
        MAloc = complexSinusT*B(:,:,t);            
        MAresponse(:,:,t) = real(MAloc.*conj(MAloc));    
        Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*MAresponse(:,r,t)./(ARresponse(:,r,t));
    end
end


% Waitbar
message = ['computing NMFARMA. iteration : 0/' int2str(Niter) ' completed'];
h = waitbar(0,message);


% iterative computation
for iter = 1:Niter

    
    % update of W
    for f0 = 1:M
        for r0 = 1:R
            vlocalv = ((Lambda(f0,:) + eps).^(beta-2).*sig2(r0,:).*reshape(MAresponse(f0,r0,:)./(ARresponse(f0,r0,:)),1,N))';
            num = V(f0,:)*vlocalv;
            denom = Lambda(f0,:)*vlocalv;
            W(f0,r0) = W(f0,r0)*(num/(denom + eps));
        end
    end

    
    if sum(isnan(W(:)))
        close(h)
        disp('W is NaN : return')
        return
    end

    
    % recomputation of Lambda (estimate of V)
    Lambda(:) = eps;

    for t = 1:N
        for r = 1:R
            Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*(MAresponse(:,r,t))./(ARresponse(:,r,t));
        end
    end

   
    % update of sigma
    for t0=1:N
        for r0=1:R
            vlocalv = (W(:,r0).*(MAresponse(:,r0,t0))./(ARresponse(:,r0,t0).*Lambda(:,t0).^(2-beta)))';
            num = vlocalv*V(:,t0);
            denom = vlocalv*Lambda(:,t0);
            sig2(r0,t0) = sig2(r0,t0)*(num/(denom + eps));
        end
    end

    if sum(isnan(sig2(:)))
        close(h)
        disp('sig2 is NaN : return')
        return;
    end
    
    % recomputation of Lambda (estimate of V)
    Lambda(:) = eps;
    for r = 1:R
        for t=1:N
            Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*(MAresponse(:,r,t))./(ARresponse(:,r,t));
        end
    end

    
    if Q>0
        % update of each MA (for each value of t and r)
        for r0=1:R
            for t0=1:N
                if sig2(r0,t0)>sig2Threshold

                    vlocal = (W(:,r0))./(ARresponse(:,r0,t0).*(Lambda(:,t0).^(2-beta)));
                    R1c = Tcolumn*(V(:,t0).*vlocal);
                    R2c = Tcolumn*(Lambda(:,t0).*vlocal);

                    % toeplitz matrix computation from column R1c and R2c (much faster than R1 = toeplitz(R1c), R2 =...)
                    R1cr = [R1c(Q+1:-1:2) ; R1c];
                    R1 = R1cr(indexToeplitzQ);
                    R2cr = [R2c(Q+1:-1:2) ; R2c];
                    R2 = R2cr(indexToeplitzQ);
                    
                   
                    B(:,r0,t0) = R2^(-1)*R1*B(:,r0,t0);
                else
                    B(:,r0,t0) = 0;
                    B(1,r0,t0) = 1;
                end
            end
        end

        % recomputation of Lambda (estimate of V)

        Lambda(:) = eps;
        for t=1:N
           Bloc = complexSinusT*B(:,:,t);
           MAresponse(:,:,t) = Bloc.*conj(Bloc);
           for r = 1:R
                Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*(MAresponse(:,r,t))./(ARresponse(:,r,t));
            end
        end
    end


    if P>0
        % update of each AR (for each value of t and r)
        for r0=1:R
            for t0=1:N
                if sig2(r0,t0)>sig2Threshold

                    vlocalv = (W(:,r0).*(MAresponse(:,r0,t0))./((ARresponse(:,r0,t0)).^2.*(Lambda(:,t0).^(2-beta))));
                    S1c = Ucolumn*(V(:,t0).*vlocalv);
                    S2c = Ucolumn*(Lambda(:,t0).*vlocalv);

                    % toeplitz matrix computation from column S1c and S2c (much faster than S1 = toeplitz(S1c)...)
                    S1cr = [S1c(P+1:-1:2) ; S1c];
                    S1 = S1cr(indexToeplitzP);
                    S2cr = [S2c(P+1:-1:2) ; S2c];
                    S2 = S2cr(indexToeplitzP);
                   
                    S1inv = S1^(-1);
                    A(:,r0,t0) = S1inv*S2*A(:,r0,t0);
                    
                else
                    A(:,r0,t0) = zeros(P+1,1);
                    A(1,r0,t0) = 1;
                end
            end
        end

        % recomputation of Lambda (estimate of V) and of filters response
        Lambda(:) = 0;
        for t=1:N
            ARloc = complexSinusU*A(:,:,t);            
            ARresponse(:,:,t) = real(ARloc.*conj(ARloc));
            for r=1:R
                Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*MAresponse(:,r,t)./(ARresponse(:,r,t));
            end
        end

    end

    
    if sum(isnan(A(:)))
        close(h)
        disp('A is NaN : return')
        return
    end
   
    % normalization/stabilization
    for r0=1:R
        for t0=1:N
            % stabilization
            if P>0
                if ~stabilityCheck(A(:,r0,t0))
                    A(:,r0,t0) = minphase(A(:,r0,t0));
                end
            end
            if Q>0
                if ~stabilityCheck(B(:,r0,t0))
                    B(:,r0,t0) = minphase(B(:,r0,t0));
                end
            end

            % normalization of filters
            if powerFiltersNormalization
                GB = sqrt(sum(MAresponse(:,r0,t0)));
                GA = 1/sqrt(sum(1./ARresponse(:,r0,t0)));
            else
                GB = B(1,r0,t0);
                GA = A(1,r0,t0);
            end


            sig2(r0,t0) = sig2(r0,t0)*abs(GB/GA).^2;
            A(:,r0,t0) = A(:,r0,t0)/GA;
            ARresponse(:,r0,t0) = ARresponse(:,r0,t0)/abs(GA).^2;
            B(:,r0,t0) = B(:,r0,t0)/GB;
            MAresponse(:,r0,t0) =  MAresponse(:,r0,t0)/abs(GB).^2;


        end
        % normalization of templates
        chosenNorm = 2;
        normW = norm(W(:,r0),chosenNorm);
        sig2(r0,:) = normW*sig2(r0,:);
        W(:,r0) = W(:,r0)/normW;
    end

    Lambda(:) = eps;
    for t=1:N
        for r=1:R
            Lambda(:,t) = Lambda(:,t) + sig2(r,t)*W(:,r).*MAresponse(:,r,t)./(ARresponse(:,r,t));
        end
    end
   
    message = ['computing NMFFARMA. iteration : ' int2str(iter) '/' int2str(Niter)];
    disp(message);
    waitbar(iter/Niter,h,message);
end

close(h)



function [hn] = minphase(h)
%     compute the minimum phase implementation hn of filter h
%     (if h contains the coefficients of an AR filter, it returns the stable implementation of that filter)

r = cplxpair(roots(h));

% Find roots inside & on unit circle
ra = abs(r);

iru   = find(abs(ra-1) <= 1e-8); % indices for roots on uc
irin  = find(ra-1 < -1e-8);  % indices for roots inside uc
irout = find(ra-1 >  1e-8);  % indices for roots outside uc

% Map roots outside the unit circle to inside:
cr = [r(iru) ; r(irin) ; 1./conj(r(irout))];

% compute minimmum phase filter
hn = h(1)*prod(abs(r(irout)))*poly(cr);



function [stable] = stabilityCheck(A)

N = length(A)-1; % Order of A(z)
stable = 1;      % stable unless shown otherwise
A = A(:);        % make sure it's a column vector
for i=N:-1:1
    rci=A(i+1);
    if abs(rci) >= 1
        stable=0;
        return;
    end
    A = (A(1:i) - rci * A(i+1:-1:2))/(1-rci^2);
end

function inst = isInstable(A)
r = roots(A);
if length(find(abs(r)-1 >  0))>0
    inst = true;
else
    inst = false;
end