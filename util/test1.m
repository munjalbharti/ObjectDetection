imdb1=load('data\rgb_object_detection-15-person-offset-4794.imdb-again.mat');
imdb2=load('data\rgb_object_detection-15-person-offset-888-new.imdb.mat');

filenames1={};
filenames2={};

ids1= find(imdb1.imdb.images.set==2);
ids2= find(imdb2.imdb.images.set==2);

count=0;
%for k=1:size(ids1,2)
 %   i=ids1(k);
  %  img=imdb1.imdb.images.data(:,:,:,i);
   % f=figure;
   % imshow(img);
    
%end

for k=1:size(ids1,2)
    i=ids1(k);
    name=imdb1.imdb.images.filenames{i};
    for m=1:size(ids2,2)
         n=ids2(m);
         name2=imdb2.imdb.images.filenames{n};
         if(strcmp(name,name2))
             count=count+1;
             break ;
         end 
    
    end 
    
    
    
end 



     