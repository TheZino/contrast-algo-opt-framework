function error = LCC_on_image(x, img)

    im_new = LCC(img, x.alp, x.sigs, x.sigr);

    error = calc_regression_error(im_new);
end
