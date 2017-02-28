function[centers] = mean_shift(points,count,bandwidth,iterations)
       
        centers=clusterTranslation(points,count,bandwidth,iterations);
        centers=floor(centers);
        
       
end

function [ new_set]= clusterTranslation(points ,count,bandwidth,iterations)

%figure;
%scatter(points(:,1), points(:,2));
%hold on ;
%scatter([91,20],[104,105],'r+');
%set(gca,'Ydir','reverse');
val = power(0.001,2);
max_iterations = iterations;
converged=0;
processed=zeros(size(points,1),1);
new_set= points;

for  run=1:max_iterations;
   % mapObj = containers.Map('KeyType','int32','ValueType','int32' );   
    if(converged == 0)
        converged = 1;
        for i=1:size(new_set,1)       
            if (processed(i) == 1)
                continue ;
            end 
                 m = findMean(new_set(i,:), points,count, bandwidth);                
                 
                 diff=new_set(i,:)-m;
                 norm2=diff(1,1) ^2+diff(1,2) ^2 ;
              
                 if(norm2 < val)
                        processed(i) =  1;
                 end
                   
                new_set(i,:) = m ;
               
                %ind=(floor(m(:,2)-1)) .* 256+floor(m(:,1));
                %if(mapObj.isKey(ind))                   
                 %   v= mapObj(ind);
                 %  v=v+1;
                %else
                 %   v=1;
                %end                
                %mapObj(ind)=v;  
                %scatter(m(1,1),m(1,2),'b+');
                %set(gca,'Ydir','reverse');
                %hold on ;
                
                if(converged == 1 && processed(i) == 1)
                    converged= 1;
                else 
                    converged=0;
                end
        end     
    end
end
end

function [m] = findMean(center, points,count, dist)

           diff = points - repmat(center,[size(points,1),1]) ;
           d=dist*dist ;
           norm2=diff(:,1) .^2+diff(:,2) .^2 ;
           k=find( norm2 <= d);
           points_inside=points(k,:);
           count_inside=count(k);
          % m   = mean(points_inside,1);
           m = sum(points_inside .* repmat(count_inside,1,2),1 ) / sum(count_inside,1);
       %    m=round(m);
          end 

