function error = scurve_on_image(x, img)

    im_new = scurve(img, x.gamma, x.lambda, x.a);

    error = calc_regression_error(im_new);
end
