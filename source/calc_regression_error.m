function error = calc_regression_error(img)
    
    global Mdl
    
    X = pyrunfile("extract_features.py","out",image=py.numpy.array(img));

    X = double(X);    

    [~, score] = predict(Mdl,X, 'Learners', 1:100);

    error = 1 - score(2);

end