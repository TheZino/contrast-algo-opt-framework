clear
clc

addpath('./source')

model_type = 'vgg16';

disp('Loading model...')
load(['./regressor_training/model/model_', model_type ,'_cleaned_increased.mat']);

algorithm = 'LCC'; %'GammaHist' 'GammaSCurve' 'LCC'

global Mdl

% set scores output as probability ditribution [0,1]
Mdl.ScoreTransform = 'doublelogit';


%% Bayes optimization

disp('Preparing variables...')

switch algorithm
    case 'GammaHist'
        gamma = optimizableVariable('gamma',[0.8,1],'Type','real');
        max_v = optimizableVariable('max_v',[0.9,1],'Type','real');
        min_v = optimizableVariable('min_v',[0,0.1],'Type','real');

    case 'GammaSCurve'
        gamma = optimizableVariable('gamma',[0.8,1],'Type','real');
        lambda = optimizableVariable('lambda',[0.5,1],'Type','real');
        a = optimizableVariable('a',[0.2,0.6],'Type','real');

    case 'LCC'
        al = optimizableVariable('alp',[0.2,1.8],'Type','real');
        sigs = optimizableVariable('sigs',[1,20],'Type','real');
        sigr = optimizableVariable('sigr',[5,250],'Type','real');
end

disp("Running optimizer...")

dataset = '~/Datasets/a5k/';
annotation = readtable([dataset, '/test_enh.csv']);

save_dir = ['./outputs/', algorithm, '_out/'];

if~exist(save_dir, 'dir')
    mkdir(save_dir)
end


nfiles = height(annotation);
params = {};

%% Optimization Per image

for ii =1:nfiles
    fprintf('\r\t\t\t\t')
    fprintf('\r%i / %i', ii, nfiles)
    % file = [dataset, '/img_all_adjusted_gamma_applied/', num2str(annotations_pos(ii,:).id), '.png'];
    file = [dataset, '/Adobe5K-Contrast/test_enh/', num2str(annotation(ii,:).id), '.jpg'];
    img = imread(file);

    switch algorithm
        case 'GammaHist'
            % Simple contrast
            fun = @(x)SC_on_image(x,img);
            results = bayesopt(fun,[gamma,max_v,min_v], ...
                'Verbose', 0, ...
                'AcquisitionFunctionName', 'expected-improvement-plus', ...
                'MaxObjectiveEvaluations', 30, ...
                'IsObjectiveDeterministic', true, ...
                'PlotFcn', []); %@plotMinObjective);
            im_n = simple_contrast(img, results.XAtMinObjective.gamma, results.XAtMinObjective.max_v, results.XAtMinObjective.min_v);

        case 'GammaSCurve'
            % S-CURVE
            fun = @(x)scurve_on_image(x,img);
            results = bayesopt(fun,[gamma,lambda,a], ...
                'Verbose', 0, ...
                'AcquisitionFunctionName', 'expected-improvement-plus', ...
                'MaxObjectiveEvaluations', 30, ...
                'IsObjectiveDeterministic', true, ...
                'PlotFcn', []); %@plotMinObjective);
            im_n = scurve(img, results.XAtMinObjective.gamma, results.XAtMinObjective.lambda, results.XAtMinObjective.a);

        case 'LCC'
            % LCC
            fun = @(x)LCC_on_image(x,img);
            results = bayesopt(fun,[al,sigs,sigr], ...
            'Verbose', 0, ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'MaxObjectiveEvaluations', 30, ...
            'IsObjectiveDeterministic', true,...
            'PlotFcn', []);
            im_n = LCC(img, results.XAtMinObjective.alp, results.XAtMinObjective.sigs, results.XAtMinObjective.sigr);
    end


    imwrite(im_n, [save_dir, num2str(annotation(ii,:).id), '.png'])
    params{end+1} = [cellstr(num2str(annotation(ii,:).id)), table2cell(results.XAtMinObjective)];
end

tmp = params';
params = vertcat(tmp{:});


switch algorithm
    case 'GammaHist'
        writetable(cell2table(params, 'VariableNames', {'name', 'gamma', 'max_v', 'min_v'}), './params/params_simple_contrast_perimage.csv');
    case 'GammaSCurve'
        writetable(cell2table(params, 'VariableNames', {'name', 'gamma','lambda', 'a'}), './params/params_scurve_perimage.csv');
    case 'LCC'
        writetable(cell2table(params, 'VariableNames', {'name', 'alp', 'sigs', 'sigr'}), './params/params_lcc_perimage.csv');
end
