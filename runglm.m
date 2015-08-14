%% example 1 : simple linear model y = b(2) x + b(1)

% generate synthetic data
T = 5000;
dt = 1e-3;
time = (1:T)*dt;
X = ones(T, 2); % design matrix
y = zeros(T, 1); % data being modeled
b = randn(2, 1); % true model coefficients

% covariate 1 = random walk
xs = zeros(T, 1);
xs(1) = 0;
for t = 2:T
  xs(t) = xs(t-1) + randn;
end
X(:, 2) = xs;

%simulate data
y = X*b;


d = glmdata([xs'; y'], time);

% set parameters:
p = glmparams();
p = p.addCovar('intercept', 0, [0 1], 'indicator');
% constant (= channel 0), [0 1] start/end of the interv
p = p.addCovar('x', 1, [0], 'indicator');
% channel 1, lag at 0
p.response = 2;

% fit model
m = glmodel();
m = m.fit(d,p);

% check model estimate
figure
plot(time, m.yEst, 'b', time, y, 'r--');

figure, m.plot(d,p)

%% example 2

% generate synthetic data
T = 5000;
dt = 1e-3;
time = (1:T)*dt;
Nparams = 7;
X = ones(T, Nparams); % design matrix
y = zeros(T, 1);
b = zeros(Nparams, 1); % true coefficients
b(1) = randn;

% covariate group 1 = random walk
% individual covariates are the walk states at different lags
% 1. make random walk
xs = zeros(T, 1);
xs(1) = 0;
for t = 2:T
  xs(t) = xs(t-1) + randn;
end
% 2. make function describing how y depends on lagged x
knots = [1 10 20 30];
% bs = [1 0.5 -0.2 0.1 0 0]';
bs = 10*[1 0.5 -0.2 0.1 0 0]';
b(2:end) = bs;
[lagtimes, lagcoeff] = plotspline(knots, bs);
% plot(lagtimes, lagcoeff);

% set parameters:
p = glmparams();
p = p.addCovar('intercept', 0, [0 1], 'indicator');
% p = p.addCovar('intercept', 0, [0:0.2:1], 'spline');
p = p.addCovar('x', 1, knots, 'spline');
p.response = 2;

% simulate data
burnin = knots(end);
y(1:burnin) = b(1); 
for t = burnin+1:T
  y(t) = b(1) + sum(xs(t - [lagtimes(1) : lagtimes(end)]) .* lagcoeff);
end
d = glmdata([xs y]', time);

% fit model
m = glmodel();
m = m.fit(d,p);

% check model estimate
figure
plot(time(burnin+1:end), m.yEst, 'b', time, y, 'r--');

% TODO: check the construction of design matrix 
% something seems wrong here
figure
m.plot(d,p)
subplot(212), plot(lagtimes, lagcoeff, 'r--', 'linewidth',2);

%% example 3

% generate synthetic data
T = 5000;
dt = 1e-3;
time = (1:T)*dt;
Nparams = 7;
X = ones(T, Nparams); % design matrix
y = zeros(T, 1);
b = zeros(Nparams, 1); % true coefficients
% b(1) = randn;
% b(1) = 0.001*randn;
b(1) = 0;

% covariate group 1 = self-history
% individual covariates are the states at different lags
% 1. make function describing how y depends on lagged x
knots = [1 10 20 30];
bs = [0 0 0.8 0.5 0.1 0]';

% knots = [1 5 10];
% bs = [0 0.1 0.1 -0.05 0]';


% bs = [0 0 -0.1 0.1 0 0]'; % oscillations
b(2:end) = bs;
[lagtimes, lagcoeff] = plotspline(knots, bs);
lagcoeff = 1 * lagcoeff / sum(lagcoeff);
figure, plot(lagtimes, lagcoeff);

% set parameters:
p = glmparams();
p = p.addCovar('intercept', 0, [0 1], 'indicator');
p = p.addCovar('self-history', 1, knots, 'spline');
p.response = 1;

% simulate data
burnin = knots(end);
% y(1:burnin) = b(1);
y(burnin) = b(1); 
for t = burnin+1:T
  y(t) = b(1) + sum(y(t - [lagtimes(1) : lagtimes(end)]) .* lagcoeff) + 1 * randn;
end
d = glmdata([y]', time);

% fit model
m = glmodel();
m = m.fit(d,p);
% 
% % check model estimate
figure
plot(time, y, 'b'); hold on;
% pause();
plot(time(burnin+1:end), m.yEst, 'r');
% 
% % TODO: check the construction of design matrix 
% % something seems wrong here
% % ANOTHER way to double check is to run the same estimates with indicator
% % basis. Do these estimates qualitatively match the spline estimates?
% % 
% figure
% m.plot(d,p)
% subplot(212), plot(lagtimes, lagcoeff, 'r--', 'linewidth',2);

%% example 3b (debugging intrinsic effects estimation)

% generate synthetic data
T = 5000;
dt = 1e-3;
time = (1:T)*dt;
% Nparams = 7;
Nparams = 14; % for asymmetric effects curve (below)
X = ones(T, Nparams); % design matrix
y = zeros(T, 1);
b = zeros(Nparams, 1); % true coefficients
% b(1) = 0.001*randn;
b(1) = 0;

% covariate group 1 = self-history
% individual covariates are the states at different lags
% 1. make function describing how y depends on lagged x
% % knots = [1 10 20 30];
% % bs = [0 0 0.8 0.5 0.1 0]';

% asymmetric effects curve
knots = [1 5:5:50];
bs = [0 0 0.2 0.5 0.2 0 0.2 0.6 1 0.6 0.2 0 0]';


% knots = [1 5 10];
% bs = [0 0.1 0.1 -0.05 0]';


% bs = [0 0 -0.1 0.1 0 0]'; % oscillations
b(2:end) = bs;
[lagtimes, lagcoeff] = plotspline(knots, bs);
lagcoeff = 1 * lagcoeff / sum(lagcoeff);
% figure, plot(lagtimes, lagcoeff);

% set parameters:
p = glmparams();
p = p.addCovar('intercept', 0, [0 1], 'indicator');
p = p.addCovar('self-history', 1, knots, 'spline');
p.response = 1;

% simulate data
burnin = knots(end);
% y(1:burnin) = b(1); 
y(burnin) = b(1); 
for t = burnin+1:T
  y(t) = b(1) + sum(y(t - [lagtimes(1) : lagtimes(end)]) .* lagcoeff) + randn;
end
d = glmdata([y]', time);

% fit model
m = glmodel();
m = m.fit(d,p);
% 

% plot intrinsic effects estimate against true curve
figure;
plot(lagtimes,lagcoeff, 'b','linewidth',2);
hold on;
[~, lagcoeff_est] = plotspline(knots, m.coeff(2:end));
plot(lagtimes,lagcoeff_est, 'r','linewidth',1);
% plot(lagtimes,flipud(lagcoeff_est), 'r','linewidth',1);

%% example 4 (voltage data)

% load data
global DATAPATH; DATAPATH=getenv('DATA');
[~,~,xlsInfo] = xlsread([DATAPATH '/spikeparams.xls']);
xlsInfo(1,:) = []; % drop headers -> row index corresponds to seizure number
szind = 3;
if isequal(class(szind), 'char'), szind = str2double(szind); end
patientName = xlsInfo{szind, 1};
seizureName = xlsInfo{szind, 2};
if isequal(class(xlsInfo{szind, 3}), 'char')
  badChannels = parsenumstr(xlsInfo{szind, 3});
else
  badChannels = xlsInfo{szind, 3};
end
goodChannels = setdiff(1:96, badChannels);
downsampleFactor = 8;

options.type = 'Neuroport';
sz = Seizure(patientName, seizureName, options);
z = zscore(double(sz.(options.type).Data));
d = glmdata(z', sz.Neuroport.Time);
d = d.downsample(downsampleFactor);
% d = glmdata(sz.Neuroport.Data', sz.Neuroport.Time);
% d = d.downsample(downsampleFactor);
% d.data = double(d.data);

dt_ms = round(.001 / d.dt);
T_knots = [0 1]; T_basis = 'indicator';
Q_knots = [0:50:250] * dt_ms; Q_basis = 'spline'; Q_knots(1) = 1; % for big spikes
R_knots = [0 1 5 10]  * dt_ms; R_basis = 'spline'; % for big spikes
Q = length(Q_knots); R = length(R_knots);
goodIntElec = neuroport_interior(badChannels);
N_good = length(goodIntElec);

%
i=1;
response = find(goodChannels==goodIntElec(i));
neighbors = neuroport_neighbors(goodIntElec(i));

%
p = glmparams();
p.response = response;
p = p.addCovar('intercept', 0, T_knots, T_basis);
p = p.addCovar('intrinsic', response, Q_knots, Q_basis);
p = p.addCovar('spatial1', neighbors(1), R_knots, R_basis);
p = p.addCovar('spatial2', neighbors(2), R_knots, R_basis);
p = p.addCovar('spatial3', neighbors(3), R_knots, R_basis);
p = p.addCovar('spatial4', neighbors(4), R_knots, R_basis);

m = glmodel();
m = m.fit(d, p);

%% example 4b (voltage, ECoG)

% load data
global DATAPATH; DATAPATH=getenv('DATA');
[~,~,xlsInfo] = xlsread([DATAPATH '/spikeparams.xls']);
xlsInfo(1,:) = []; % drop headers -> row index corresponds to seizure number
szind = 3;
if isequal(class(szind), 'char'), szind = str2double(szind); end
patientName = xlsInfo{szind, 1};
seizureName = xlsInfo{szind, 2};
if isequal(class(xlsInfo{szind, 3}), 'char')
  badChannels = parsenumstr(xlsInfo{szind, 3});
else
  badChannels = xlsInfo{szind, 3};
end
goodChannels = setdiff(1:96, badChannels);
downsampleFactor = 8;

options.type = 'Neuroport';
sz = Seizure(patientName, seizureName, options);
z = zscore(double(sz.(options.type).Data));
d = glmdata(z', sz.Neuroport.Time);
d = d.downsample(downsampleFactor);
% d = glmdata(sz.Neuroport.Data', sz.Neuroport.Time);
% d = d.downsample(downsampleFactor);
% d.data = double(d.data);

dt_ms = round(.001 / d.dt);
T_knots = [0 1]; T_basis = 'indicator';
Q_knots = [0:50:250] * dt_ms; Q_basis = 'spline'; Q_knots(1) = 1; % for big spikes
R_knots = [0 1 5 10]  * dt_ms; R_basis = 'spline'; % for big spikes
Q = length(Q_knots); R = length(R_knots);
goodIntElec = neuroport_interior(badChannels);
N_good = length(goodIntElec);

%
i=1;
response = find(goodChannels==goodIntElec(i));
neighbors = neuroport_neighbors(goodIntElec(i));

%
p = glmparams();
p.response = response;
p = p.addCovar('intercept', 0, T_knots, T_basis);
p = p.addCovar('intrinsic', response, Q_knots, Q_basis);
p = p.addCovar('spatial1', neighbors(1), R_knots, R_basis);
p = p.addCovar('spatial2', neighbors(2), R_knots, R_basis);
p = p.addCovar('spatial3', neighbors(3), R_knots, R_basis);
p = p.addCovar('spatial4', neighbors(4), R_knots, R_basis);

m = glmodel();
m = m.fit(d, p);

%% example 5 (hierarchy of models vs correlation)

% load data
global DATAPATH; DATAPATH=getenv('DATA');
[~,~,xlsInfo] = xlsread([DATAPATH '/spikeparams.xls']);
xlsInfo(1,:) = []; % drop headers -> row index corresponds to seizure number
szind = 1;
if isequal(class(szind), 'char'), szind = str2double(szind); end
patientName = xlsInfo{szind, 1};
seizureName = xlsInfo{szind, 2};
if isequal(class(xlsInfo{szind, 3}), 'char')
  badChannels = parsenumstr(xlsInfo{szind, 3});
else
  badChannels = xlsInfo{szind, 3};
end
goodChannels = setdiff(1:96, badChannels);
downsampleFactor = 8;

options.type = 'Neuroport';
sz = Seizure(patientName, seizureName, options);
z = zscore(double(sz.(options.type).Data));
d = glmdata(z', sz.Neuroport.Time);
d = d.downsample(downsampleFactor);
% d = glmdata(sz.Neuroport.Data', sz.Neuroport.Time);
% d = d.downsample(downsampleFactor);
% d.data = double(d.data);

dt_ms = round(.001 / d.dt);
T_knots = [0 1]; T_basis = 'indicator';
Q_knots = [0:50:250] * dt_ms; Q_basis = 'spline'; Q_knots(1) = 1; % for big spikes
R_knots = [0 1 5 10]  * dt_ms; R_basis = 'spline'; % for big spikes
Q = length(Q_knots); R = length(R_knots);
% goodIntElec = neuroport_interior(badChannels);
goodIntElec = [5 7];
Ngood = length(goodIntElec);

% % partition seizures into windows
% szStart = xlsInfo{szind, 4}; % for LADs, beginning 10s of sec into sz
szStart = 0; % begin at sz onset
% szEnd = xlsInfo{szind, 5};
szEnd = 10;
windowSize = 10;
startTimes = szStart : szEnd-windowSize;
endTimes = startTimes + windowSize;
Nwindows = length(startTimes);

% % initialize parameters
p = glmparams();
response = goodIntElec(1);
neighbors = neuroport_neighbors(response);
p.response = response;
p = p.addCovar('intercept', 0, T_knots, T_basis);
p = p.addCovar('intrinsic', response, Q_knots, Q_basis);
p = p.addCovar('spatial1', neighbors(1), R_knots, R_basis);
p = p.addCovar('spatial2', neighbors(2), R_knots, R_basis);
p = p.addCovar('spatial3', neighbors(3), R_knots, R_basis);
p = p.addCovar('spatial4', neighbors(4), R_knots, R_basis);

% % initialize arrays for storing auto-corr, cross-corr
Nlagsac = Q_knots(end);
Nlagsxc = R_knots(end);
ac = zeros(Ngood,Nlagsac+1,Nwindows);
xc = zeros(Ngood,2*Nlagsxc+1,Nwindows,4);

% % initialize arrays for storing hierarchy of models
msNull = cell(1,Nwindows);
msInt = cell(1,Nwindows);
msSpace = cell(1,Nwindows);
msFull = cell(1,Nwindows);

fprintf('\nLoaded data, initialized arrays.\n');

% % estimate models, correlations
for n = 1:Nwindows
  n
  % make concatenated design matrix X and data y
  m = glmodel();
  d0 = d.subtime(startTimes(n),endTimes(n));
  %T0 = d0.T-p.getBurnIn();
  NT = Ngood * d0.T;
  X = zeros(NT, p.Ncovar());
  y = zeros(NT, 1);  
  for i = 1:Ngood
%     i
    response = goodIntElec(i);
    neighbors = neuroport_neighbors(response);
    % covar #1 is intercept -- channel = 0 by convention
    p.covarChannels{2} = response;
    p.covarChannels{3} = neighbors(1);
    p.covarChannels{4} = neighbors(2);
    p.covarChannels{5} = neighbors(3);
    p.covarChannels{6} = neighbors(4);    
    m = m.makeX(d0,p);
    X((i-1)*d0.T+(1:d0.T),:) = m.X;
    y((i-1)*d0.T+(1:d0.T)) = d0.data(response,:);
    ac(i,:,n) = autocorr(d0.data(response,:),Nlagsac);
    xc(i,:,n,1) = xcorr(d0.data(response,:),d0.data(neighbors(1),:),Nlagsxc,'coef');
    xc(i,:,n,2) = xcorr(d0.data(response,:),d0.data(neighbors(2),:),Nlagsxc,'coef');
    xc(i,:,n,3) = xcorr(d0.data(response,:),d0.data(neighbors(3),:),Nlagsxc,'coef');
    xc(i,:,n,4) = xcorr(d0.data(response,:),d0.data(neighbors(4),:),Nlagsxc,'coef');
  end
  
  % estimate firing properties by max likelihood 
  fprintf('\nDesign matrix built. Estimating hierarchy of models...\n');
  % null model (one term)
  m.X = X(:,1);
  m.y = y;
  m = m.irls(); % iteratively reweighted least squares estimate
  m.X = [];
  m.y = [];
  msNull{n} = m;
  
  % intrinsic effects (self-history) model
  m.X = X(:,[p.covarInd{1:2}]);
  m.y = y;
  m = m.irls(); % iteratively reweighted least squares estimate
  m.X = [];
  m.y = [];
  msInt{n} = m;
  
  % spatial effects (ensemble-history) model
  m.X = X(:,[p.covarInd{[1,3:6]}]);
  m.y = y;
  m = m.irls(); % iteratively reweighted least squares estimate
  m.X = [];
  m.y = [];
  msSpace{n} = m;
  
  % full model (intrinsic + spatial)
  m.X = X;
  m.y = y;
  m = m.irls(); % iteratively reweighted least squares estimate
  m.X = [];
  m.y = [];
  msFull{n} = m;
  
  save ~/testGLM.mat p msFull msNull msInt msSpace xc ac startTimes endTimes szind
  
  fprintf(['Done for window [' num2str(startTimes(n)) ', ' num2str(endTimes(n)) ']\n']);
end
% save ~/testGLM.mat p msFull msNull msInt msSpace xc ac startTimes endTimes szind

