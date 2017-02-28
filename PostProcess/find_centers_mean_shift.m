function [ centers_y,centers_x,contri ] = find_centers_mean_shift( offset, threshold, bandwidth, iterations )
%FIND_CENTERS_MEAN_SHIFT Summary of this function goes here
%   Detailed explanation goes here
        tStart=tic ;
        [center_mask,center_contri]=centers_heat_map( offset, [1,1] );
      %  f1=figure;
      %  imagesc(center_mask);

        %thresholding
        th_center_mask=center_mask;
        th_center_mask(find(th_center_mask <= threshold)) = 0;

        [y,x,v]=find(th_center_mask);
       % points_bak=[];
        
       
        if(size(y,1)==0)
            centers_y=[];
            centers_x=[];
            contri=[];
          %  close(f1);
            return ;
        end 
        
       % for k=1:size(y,1)
        %     points_bak=[points_bak; repmat([y(k),x(k)],[v(k),1] )  ];
       % end

        points=[y,x];
       %centers and points have same size
         
        centers=mean_shift(points,v,bandwidth,iterations);

    %    centers_count=accumarray(centers,1);
     %   th=50;
      %  [centers_y, centers_x]=find(centers_count > th);
        centers_y= centers(:,1);
        centers_x= centers(:,2);
        contri = struct('x_pos',[],'y_pos',[]);
        
        for i=1:size(centers,1)
             y=centers_y(i);
             x=centers_x(i);
             contri(i) = struct('x_pos',[],'y_pos',[]);
             
             contri_points=unique(points(find(centers(:,1)==y & centers(:,2)==x),:),'rows');
             
             
             for k=1:size(contri_points,1)
                 contri(i).x_pos =[contri(i).x_pos; center_contri(contri_points(k,1),contri_points(k,2)).x_pos];
                 contri(i).y_pos =[contri(i).y_pos; center_contri(contri_points(k,1),contri_points(k,2)).y_pos];
           
             end 
             
             contri(i).x_pos=[contri(i).x_pos; center_contri(y,x).x_pos];
             contri(i).y_pos=[contri(i).y_pos; center_contri(y,x).y_pos];   
        end 

          tElapsed = toc(tStart);
          fprintf('\nMeanshift %f sec for image %d X %d\n',tElapsed,size(offset,1),size(offset,2));
           
     %   hold on ;
     %   plot(centers_x,centers_y,'b+');

      %  close(f1);
end

