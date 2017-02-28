function  err = ErrorFn(opts, labels, res )
%ERRORFN Summary of this function goes here
%   Detailed explanation goes here
    predictions = gather(res(end-1).x) ; 
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions(1,1,1,:), reshape(labels,1,1,1,[]));
    err = sum(error(:)) ;
end

