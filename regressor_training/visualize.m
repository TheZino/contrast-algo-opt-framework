clear
clc

%% Inference on data

model_type = 'vgg16';

disp('Loading model');
load(['./model/model_', model_type ,'_cleaned_increased_2.mat']);
dataset = '~/Datasets/a5k/';


disp('Loading annotations');

annotation = readtable([dataset, '/test.csv']);
image_id = annotation.id;


feat_dir = './features/';

features_file = [feat_dir, '/features_', model_type ,'_test.mat'];


if isfile(features_file)
    disp('Loading features');
    load(features_file)

    if ~strcmp(model_type,'HC')
        X = X_net;
    end


else
    nfiles = length(image_id);
    labels = [];
    fprintf('Extracting features:\n')
    for ii=1:nfiles
    
        file = [dataset, '/img_all_adjusted_gamma_applied/', num2str(image_id(ii)), '.png'];
        fprintf('\r\t\t\t')
        fprintf('\r%s', [num2str(image_id(ii)), '.png'])
        img = imread(file);
        features = feature_extraction(img);
   
    end
end





image_ids = unique(annotation.img_number);

image_ids= image_ids(1:8);

Mdl.ScoreTransform = 'doublelogit';

scores = {};

disp(image_ids)

X_or = X;

for jj = 1:length(image_ids)

	fprintf("\r\t\t\t\t\t");
	fprintf("\r %i / %i", jj, length(image_ids));

	mask1 = ismember(annotation.img_number, image_ids(jj));
	mask2 = ismember(annotation.label, 1);
	X = X_or(mask1, :);

	nfeat = size(X,1);
	
	[label, score] = predict(Mdl,X, 'Learners', 1:100);
	Y_calc = double(score(:,2));
	

	annot = annotation(mask1,:);

	[mx, idx] = max(Y_calc);

	im = imread([dataset, '/img_all_adjusted_gamma_applied/', num2str(annot(idx,:).id), '.png']);

 	if ~exist(['./plot/' model_type, '/'], 'dir')
       mkdir(['./plot/' model_type, '/']);
    end

	% imwrite(im, ['./plot/' model_type, '/', num2str(image_ids(jj)), '.png'])


	% f = figure('visible','off');
	f=figure();
	coord = annotation(mask1,4:5);
	[xi,yi] = meshgrid(min(coord{:,1}):0.01:max(coord{:,1}), min(coord{:,2}):0.01:max(coord{:,2}));
	zi = griddata(coord{:,1},coord{:,2},Y_calc,xi,yi);
	s = surf(xi,yi,zi);
	s.EdgeColor = 'none'
	% view(0,90)
	% exportgraphics(f,['./', num2str(image_ids(jj)) '.png'],'Resolution',300)


end


