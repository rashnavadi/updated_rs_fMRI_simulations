%% July 2024, By Tahereh Rashnavadi


close all; clear;
Basefile= '/Users/trashnavadi/Documents/Data_Analysis/2022/analyses/kmeans_investigation/2024/July/final';

% Initialize variables
% Number of iterations
numIterations = 10;
nSubj = numIterations;
TR = 2; % in seconds
nstates = 2;
k = nstates;
% WSize = [30, 40, 50]; % in timepoints (TR = 2sec)
WSize = 50; % in timepoints (TR = 2sec)

% transitionPoint = [15, 45, 75, 120]; % in TR
transitionPoint = 75;
tc_length = 150;
nTimePts = tc_length;

% noise_level = [400, 1000];
noise_level = 400.0;
scale_tc = 25.0;

t=0:TR:(TR*tc_length - TR);
time = t;

x=sin(2*pi*t*0.06);
x = x * scale_tc;
x=x'+ 10000;

y_low=sin(2*pi*(t(1:transitionPoint)*0.06));
y_high=sin(2*pi*(t(transitionPoint+1:tc_length)*0.06+0.15));
y=[y_low,y_high];
y = y * scale_tc;
y=y'+10000;

z_low=sin(2*pi*(t(1:transitionPoint)*0.06+0.15));
z_high = sin(2 * pi * (t(transitionPoint+1:tc_length)*0.06));
z = [z_low, z_high];
z = z * scale_tc;
z = z'+10000;

x_ylow_r = corrcoef(x(1:transitionPoint),y(1:transitionPoint));
x_yhigh_r = corrcoef(x(transitionPoint+1:tc_length),y(transitionPoint+1:tc_length));
x_zlow_r = corrcoef(x(1:transitionPoint),z(1:transitionPoint));
x_zhigh_r = corrcoef(x(transitionPoint+1:tc_length),z(transitionPoint+1:tc_length));
y_zlow_r = corrcoef(y(1:transitionPoint),z(1:transitionPoint));
y_zhigh_r = corrcoef(y(transitionPoint+1:tc_length),z(transitionPoint+1:tc_length));

% Preallocate arrays to store state flips
StateFlip_SWC = cell(numIterations, 1);
StateFlip_HOCo = cell(numIterations, 1);
% Initialize storage for noisy time series
all_noisy_x = zeros(nTimePts, numIterations);
all_noisy_y = zeros(nTimePts, numIterations);
all_noisy_z = zeros(nTimePts, numIterations);

% desired_snr = 50;

% Run the function for 1000 iterations
for iter = 1:numIterations
    px = noise_level * pinknoise(0.5 * 2 * tc_length);
    py = noise_level * pinknoise(0.5 * 2 * tc_length);
    pz = noise_level * pinknoise(0.5 * 2 * tc_length);

    v_px = var(px);
    v_py = var(py);
    v_pz = var(pz);

    xp = x + px;
    xph = highpass(xp, .01, .5);

    yp = y + py;
    yph = highpass(yp, .01, .5);

    zp = z + pz;
    zph = highpass(zp, .01, .5);

    rxya = corrcoef(xp(1:transitionPoint),yp(1:transitionPoint));
    rxyb = corrcoef(xp(transitionPoint:tc_length),yp(transitionPoint:tc_length));
    rxza = corrcoef(xp(1:transitionPoint),zp(1:transitionPoint));
    rxzb = corrcoef(xp(transitionPoint:tc_length),zp(transitionPoint:tc_length));
    ryza = corrcoef(yp(1:transitionPoint),zp(1:transitionPoint));
    ryzb = corrcoef(yp(transitionPoint:tc_length),zp(transitionPoint:tc_length));

    rxyah = corrcoef(xph(1:transitionPoint),yph(1:transitionPoint));
    rxybh = corrcoef(xph(transitionPoint:tc_length),yph(transitionPoint:tc_length));
    rxzah = corrcoef(xph(1:transitionPoint),zph(1:transitionPoint));
    rxzbh = corrcoef(xph(transitionPoint:tc_length),zph(transitionPoint:tc_length));
    ryzah = corrcoef(yph(1:transitionPoint),zph(1:transitionPoint));
    ryzbh = corrcoef(yph(transitionPoint:tc_length),zph(transitionPoint:tc_length));

%     figure; hold on; plot(xph); plot(yph); plot(zph);

    % Store the noisy signals
    all_noisy_x(:, iter) = xph;
    all_noisy_y(:, iter) = yph;
    all_noisy_z(:, iter) = zph;

    % Prepare data for HOCo and SWC
    TSs = cell(2, 3);
    TSs{1, 1} = 'timeseries_1';
    TSs{2, 1} = all_noisy_x(:, iter);
    TSs{1, 2} = 'timeseries_2';
    TSs{2, 2} = all_noisy_y(:, iter);
    TSs{1, 3} = 'timeseries_3';
    TSs{2, 3} = all_noisy_z(:, iter);
    save(fullfile(Basefile, 'sim_timeseries_1.mat'), 'TSs')

    % HOCo and SWC processing
    load(fullfile(Basefile, 'sim_timeseries_1.mat'))
    SaveSuffix = 'Pearsons'; SurrOption = 0;
    Time_series_file = fullfile(Basefile, 'sim_timeseries_1.mat');
    hoco_FunConn(Time_series_file, WSize, SurrOption, SaveSuffix)
    load(fullfile(Basefile, 'mtx_SldWFCPearsons.mat'))
    FC_21 = SldWFCArray{3, 2};
    FC_31 = SldWFCArray{4, 2};
    FC_32 = SldWFCArray{4, 3};

    % SWC Analysis
    kmeans_data_SWC = [FC_21, FC_31, FC_32];
    [StateIdx_SWC, Centroid_SWC, SumDistPartial_SWC] = kmeans(kmeans_data_SWC, k, 'Distance', 'cityblock', 'MaxIter', 100, 'Replicates', 100);
    StateFlip_SWC{iter, 1} = find(logical(diff(StateIdx_SWC)));

    % HOCo Analysis
    X_full = TSs{2, 1};
    Obs_full = TSs{2, 2};
    T_full = TSs{2, 3};
    param11 = 1; param12 = 1; param21 = 1;

    [static12, dynamic12] = hoco_fbglmfit(X_full, Obs_full, WSize, param11, param12, param21, TR);
    [static13, dynamic13] = hoco_fbglmfit(X_full, T_full, WSize, param11, param12, param21, TR);
    [static23, dynamic23] = hoco_fbglmfit(Obs_full, T_full, WSize, param11, param12, param21, TR);

    % ========================================
    %     % before running kmeans, trim the HOCo results, remove the FC matrices obtained from
    % shrinking/growing windows , because the truncated HOCo results were
    % better than the HOCo itself
    dynamic12.bb  = dynamic12.bb(WSize/2 : size(X_full) - WSize/2);
    dynamic13.bb  = dynamic13.bb(WSize/2 : size(X_full) - WSize/2);
    dynamic23.bb  = dynamic23.bb(WSize/2 : size(X_full) - WSize/2);

    kmeans_data_HOCo = [nonzeros(dynamic12.bb), nonzeros(dynamic13.bb), nonzeros(dynamic23.bb)];
    [StateIdx_HOCo, Centroid_HOCo, SumDistPartial_HOCo] = kmeans(kmeans_data_HOCo, k, 'Distance', 'cityblock', 'MaxIter', 100, 'Replicates', 100);
    StateFlip_HOCo{iter, 1} = find(logical(diff(StateIdx_HOCo)));
end
    % Save the results for each noise level
    save(fullfile(Basefile, ['timeseries_with_noise_level_' num2str(noise_level) '.mat']), 'all_noisy_x', 'all_noisy_y', 'all_noisy_z', 'StateFlip_HOCo', 'StateFlip_SWC');




