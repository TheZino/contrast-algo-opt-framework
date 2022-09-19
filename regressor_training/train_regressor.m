clear
clc

addpath('../')
dataset = '~/Datasets/a5k/';
model_type = 'vgg16';

%% Image annotations
% img_no: Image number.
%         Refer to the MIT-Adobe FiveK Dataset for the actual images.
% label: Acceptability provided by the Amazon Mechanical Turkers.
%        1 indicates acceptable adjustment, while 0 indicates not acceptable.
% brightness: Brightness setting, refer to our paper for detail.
% contrast: Contrast settings, refer to our paper for detail.
% id: Id of the data points.
%     Use this to refer to the pre-adjusted images (see below).

disp('Loading annotations');
ann_or = readtable([dataset, '/acceptable_adj.csv']);
annotation = readtable([dataset, '/train_cleaned_increased.csv']);
image_id = annotation.id;
Y = annotation.label;

%% Feature extraction 
% extract for each image in the dataset the sequence of features and
% prepare the data for ensamble fitting

feat_dir = '/home/zino/Projects/ContrastOptimization/regressor_training/features/';
features_file = [feat_dir, '/features_', model_type,'_train_cleaned_increased.mat'];

if isfile(features_file)
    
    disp('Loading features');
    load(features_file)
    
else
    
    nfiles = length(image_id);
    X = [];

    fprintf('Extracting features:\n')
    for ii=1:nfiles

       file = [dataset, '/img_all_adjusted_gamma_applied/', num2str(image_id(ii)), '.png'];
       fprintf('\r\t\t\t')
       fprintf('\r%i / %i', ii, nfiles)
       img = imread(file);
       features = feature_extraction(img);

       X = [X; features];

    end

    save(features_file, 'X');

end


%% Training Adaptive Logistic Regression
% Fit the logitboost using the adob5k preprocessed dataset and using the
% reduced feature extractor written from scratch

if ~strcmp(model_type, 'HC')
    X = squeeze(X_net);
    % mask = ismember(ann_or.img_number, unique(annotation.img_number));
    % X = X(mask, :);
end

disp('Training Logistic Regressor')
rng('default')
t= templateTree('Reproducible',true);

Mdl = fitcensemble(X, Y, 'Learner', {templateTree('MaxNumSplits',10), ...
                            templateTree('MaxNumSplits',10), ...
                            templateTree('MaxNumSplits',10), ...
                            templateTree('MaxNumSplits',10), ...
                            templateTree('MaxNumSplits',10)});
    
    % 'Method', 'LogitBoost', 'Learner', t);
    % 'Method', 'LogitBoost', 'Learner', t, 'LearnRate',0.5);
    % 'Method', 'LogitBoost','OptimizeHyperparameters',...
        % {'NumLearningCycles','LearnRate','MaxNumSplits'});
        % 'NumLearningCycles', 100,...
        %     


disp('Model saving')
save(['./model/model_', model_type, '_cleaned_increased.mat'], 'Mdl', '-v7.3')

