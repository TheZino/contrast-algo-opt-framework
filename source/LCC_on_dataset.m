function error = LCC_on_dataset(x)

    dataset = '~/Datasets/a5k/';
    annotation = readtable([dataset, '/train.csv']);
    image_id = annotation.id;
    nfiles = length(image_id);
    errors = [];
    for ii=1:nfiles
        fprintf('\r\t\t\t')
        fprintf('\r%i / %i', ii, nfiles)

        file = [dataset, '/img_all_adjusted_gamma_applied/', num2str(image_id(ii)), '.png'];
        img = imread(file);

        im_new = LCC(img, x(1), x(2), x(3));

        errors = [errors; calc_regression_error(im_new)];
    end

    error = norm(errors,2);
end