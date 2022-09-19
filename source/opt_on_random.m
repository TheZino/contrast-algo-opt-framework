function error = opt_on_random(x, algorithm)

    dataset = '~/Datasets/a5k/';
    annotation = readtable([dataset, '/train.csv']);
    image_id = annotation.id;
    
    msize = numel(image_id);
    image_id = image_id(randperm(msize, 120));

    nfiles = length(image_id);
    errors = [];
    for ii=1:nfiles
        fprintf('\r\t\t\t\t')
        fprintf('\r%i / %i', ii, nfiles)

        file = [dataset, '/img_all_adjusted_gamma_applied/', num2str(image_id(ii)), '.png'];
        img = imread(file);

        switch algorithm
            case 'GammaHist'
                im_new = simple_contrast(img, x.gamma, x.max_v, x.min_v);
            case 'GammaSCurve'
                im_new = scurve(img, x.gamma, x.lambda, x.a);
            case 'LCC'
                im_new = LCC(img, x.alp, x.sigs, x.sigr);
            end

        errors = [errors; calc_regression_error(im_new)];
    end

    error = norm(errors,2);
end