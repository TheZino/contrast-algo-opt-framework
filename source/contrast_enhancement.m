clear
clc


dataset = '~/Datasets/a5k/';

params = readtable('./params/params.csv');

nfiles = height(params)

save_dir = './scurve_out/'

if~exist(save_dir, 'dir')
    mkdir(save_dir)
end

for ii =1:nfiles
    fprintf('\r\t\t\t\t')
    fprintf('\r%i / %i', ii, nfiles)
    
    file = [dataset, '/test_enh/', num2str(params.name(ii)), '.jpg'];
    
    img = imread(file);

    % img_out = simple_contrast(img, params(ii,:).gamma, params(ii,:).max_v, params(ii,:).min_v);
    img_out = scurve(img, params(ii,:).lambda, params(ii,:).a);

    imwrite(img_out, [save_dir, num2str(params.name(ii)), '.png']);
end
