function []= displayImage(im, lb, pred)
% -------------------------------------------------------------------------
            subplot(1,3,1) ;
            image(im) ;
            axis image ;
            title('source image') ;

            subplot(1,3,2) ;
            image(uint8(lb-1)) ;
            axis image ;
            title('ground truth')

            cmap = labelColors(21) ;
            subplot(1,3,3) ;
            image(uint8(pred-1)) ;
            axis image ;
            title('predicted') ;

            colormap(cmap) ;

            
            
end 

