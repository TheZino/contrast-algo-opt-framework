function error = SC_on_image(x, img)

    im_new = simple_contrast(img, x.gamma, x.max_v, x.min_v);

    error = calc_regression_error(im_new);
end
