function im_new = simple_contrast(img, gamma, max_v, min_v)

	img = im2double(img);

	im_new = img .^ gamma;

	im_new = imadjust(im_new, [min_v, max_v]);

end