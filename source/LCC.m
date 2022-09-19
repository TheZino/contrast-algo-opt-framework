function im_new = LCC(img, alpha, sigmas, sigmar)
    
    img= im2double(img);
    im = rgb2ycbcr(img);
    y = im(:,:,1);

% A pixel is considered dark if its luminance Y is less than 35
% and its chroma radius [(Cb − 128)2 + (Cr− 128) 2] 1 / 2 is less than 20.

    % dark pixels percentage
    mask = im(:,:,1)<35 & ((im(:,:,2) - 0.5).*2 + (im(:,:,3) - 0.5).*2)/2 < 20;

    dark_pixels = sum(mask(:));

%     [hh,ww] = size(y);
%     sigmas = sqrt(hh^2 + ww^2) * 0.02;
%     sigmar = mean(imgradient(y), 'all');

    % fast bilateral filter
    % simgas [1, 20]
    % sigmar [5, 250]
    mask = shiftableBF(y, sigmas, sigmar); 
%     mask = imbilatfilt(y, sigmar, sigmas);
    mask = 1-mask;

    % original alpha
%     mean_intensity = mean(y, 'all');
%     if mean_intensity < 0.5
%         alpha = log(mean_intensity) / log(0.5);
%     else
%         alpha = log(0.5) / log(mean_intensity);
%     end

%     alpha = 2.5;
    
    gamma = alpha .^ ((0.5 - mask) / 0.5);
    y_new = y .^ gamma;

    % Contrast (Stratching and Clipping)

    if dark_pixels > 0
        
        [ipixelCount, ~] = imhist(y, 256);
        cdf = cumsum(ipixelCount);
        [~,idx] = min(abs(cdf-(dark_pixels*0.3)));
        b_input30 = idx;

        [ipixelCount, ~] = imhist(y_new, 256);
        cdf = cumsum(ipixelCount);
        [~,idx] = min(abs(cdf-(dark_pixels*0.3)));
        b_output30 = idx;

        bstr = (b_output30 - b_input30);
    else
        bstr = floor(quantile(reshape(y_new, [], 1), 0.002) * 255);
    end

    if bstr > 50
            bstr = 50;
    end

    dark_bound = bstr/2^8;


    bright_b = floor(quantile(y_new(:), 1-0.002) * 255);

    if (255 - bright_b) > 50
        bright_b = 255 - 50;
    end

    bright_bound = bright_b / 2^8;

    if dark_bound<0
        dark_bound = 0;
    end
    if bright_bound>1
        bright_bound = 1;
    end


%     y_new = (y_new - dark_bound) ./ (bright_bound - dark_bound);
    y_new = imadjust(y_new, [dark_bound, bright_bound], []);

    im(:,:,1) = y_new;
    im_new = ycbcr2rgb(im);
    
    % Saturation

    im_tmp = img;

    r = im_tmp(:,:,1);
    g = im_tmp(:,:,2);
    b = im_tmp(:,:,3);

    r_new = 0.5.*( ((y_new./y) .* (r + y)) + r - y );
    g_new = 0.5.*( ((y_new./y) .* (g + y)) + g - y );
    b_new = 0.5.*( ((y_new./y) .* (b + y)) + b - y );

    im_new(:,:,1) = r_new;
    im_new(:,:,2) = g_new;
    im_new(:,:,3) = b_new;

end
