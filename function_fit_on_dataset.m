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
        gamma = optimizableVariable('gamma',[0.6,1],'Type','real');
        max_v = optimizableVariable('max_v',[0.9,1],'Type','real');
        min_v = optimizableVariable('min_v',[0,0.1],'Type','real');
        pars = [gamma,max_v,min_v];

    case 'GammaSCurve'
        gamma = optimizableVariable('gamma',[0.8,1],'Type','real');
        lambda = optimizableVariable('lambda',[0.5,1],'Type','real');
        a = optimizableVariable('a',[0.2,0.6],'Type','real');
        pars = [gamma,lambda,a];

    case 'LCC'
        al = optimizableVariable('alp',[0.2,1.8],'Type','real');
        sigs = optimizableVariable('sigs',[1,20],'Type','real');
        sigr = optimizableVariable('sigr',[5,250],'Type','real');
        pars = [al, sigs, sigr];
end


disp("Running optimizer...")


save_dir = ['./outputs/', algorithm, '_out/'];

if~exist(save_dir, 'dir')
    mkdir(save_dir)
end


%% Optimization on dataset

% LCC
fun = @(x)opt_on_random(x, algorithm);
results = bayesopt(fun,pars, ...
'Verbose', 0, ...
'AcquisitionFunctionName', 'expected-improvement-plus', ...
'MaxObjectiveEvaluations', 30, ...
'IsObjectiveDeterministic', true,...
'PlotFcn', []);
params = results.XAtMinObjective;



switch algorithm
    case 'GammaHist'
        writetable(cell2table(params, 'VariableNames', {'name', 'gamma', 'max_v', 'min_v'}), './params/params_simple_contrast_dataset.csv');
    case 'GammaSCurve'
        writetable(cell2table(params, 'VariableNames', {'name', 'gamma','lambda', 'a'}), './params/params_scurve_dataset.csv');
    case 'LCC'
        writetable(cell2table(params, 'VariableNames', {'name', 'alp', 'sigs', 'sigr'}), './params/params_lcc_dataset.csv');
end

