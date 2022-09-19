function img_new = scurve(img, gamma, lambda, a)

	img = im2double(img);
	
	img = img.^gamma;

	mask = img <= a;
	img_new = img;
	img_new(mask) = a - a*(1 - img(mask)/a).^lambda;
	img_new(~mask) = a + (1-a)*((img(~mask)-a) / (1-a)).^lambda;

end