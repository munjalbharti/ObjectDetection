%% Demo for the CocoApi (see CocoApi.m)
close all;
clear ;
 
%% initialize COCO api (please specify dataType/annType below)
annTypes = { 'instances'};
dataType='train2014'; annType=annTypes{1}; % specify dataType/annType
annFile=sprintf('../annotations/%s_%s.json',annType,dataType);
coco=CocoApi(annFile);
 

 
%% get all images containing given categories, select one at random
 
%Voc classes changed to  ms coco in same order
 
 opts.classes={...
    'airplane' 
    'bicycle' 
    'bird' 
    'boat' 
    'bottle' 
    'bus' 
    'car' 
    'cat' 
    'chair' 
    'cow' 
    'dining table' 
    'dog' 
    'horse' 
    'motorcycle' 
    'person' 
    'potted plant' 
    'sheep' 
    'sofa'
    'train' 
    'tv' };
 

 %%{   
opts.nclasses=length(opts.classes);
opts.voc_imagespath_int='../train_images_intst_cat/';

%%}
 
%{
opts.voc_imagespath_int='../train_images/';

%}

mkdir(opts.voc_imagespath_int);
catIds = coco.getCatIds('catNms', opts.classes);
imgIds = coco.getImgIds('catIds',catIds);

 
 
tic;
sum_all1=0;
sum_all2=0;


       
 
total_imgs=length(imgIds);
for k=1:total_imgs
    %% load and display image
    fid = fopen(fullfile(opts.voc_imagespath_int,'train_perct.csv'),'at');
     if toc >100
            fprintf('Status %d/%d\n',k,length(imgIds));            
            tic;
     end
        
    imgId = imgIds(k);
   %imgId=294;
    img = coco.loadImgs(imgId);
    
    annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[0]);
    anns = coco.loadAnns(annIds); 
    
    I = imread(sprintf('../images/%s/%s',dataType,img.file_name));
    f1=figure('visible','off'); imagesc(I); 
    axis('image'); 
    set(gca,'XTick',[],'YTick',[])
    coco.showAnns(anns);
    all_masks=coco.getSegmentationMasks(anns,img);
    i_mask= zeros(img.height,img.width,'logical');
    u_mask= zeros(img.height,img.width,'logical');
    n=size(all_masks,3)  ;
        
    u_mask(find(sum(all_masks(:,:,:),3) > 0))=1;
    
    
    %{
        
        
        for p=1:n
            c_mask=all_masks(:,:,p);  
            all_masks_exc=all_masks(:,:,setdiff([1:n],p));
            
            u_w_mask= zeros(img.height,img.width,'logical');
            tmp=sum(all_masks_exc(:,:,:),3);
            u_w_mask(find(tmp > 0))=1;

            mask=c_mask & u_w_mask ;
            i_mask= i_mask | mask;        
        end   
   
 
    %}
    
   
        for p=1:n
            c_mask = all_masks(:,:,p);  
            
            cat_id= setdiff(unique(c_mask),0);  
            
            if(isempty(cat_id))
                continue ;
            end 
            
            ind = find(all_masks == cat_id);
            [~,~,z]=ind2sub(size(all_masks),ind);
            z=unique(z);
            %z=p;  %before
            
            all_masks_exc= all_masks(:,:,setdiff([1:n],z));
            if(size(all_masks_exc,3) ~= 0)   
                     u_w_mask= zeros(img.height,img.width,'logical');
                     tmp=sum(all_masks_exc(:,:,:),3);
                     u_w_mask(find(tmp > 0))=1;

                     mask=c_mask & u_w_mask ;
                     i_mask= i_mask | mask;
            end 
        end   
 
       
       total_object_pixels= length(find(u_mask));
       total_img_pixels=img.height*img.width;
       total_ints_pixels=length(find(i_mask));
       
       if(total_object_pixels ==0)
          disp('STOP');
       end 
       
       per_int_pi1=total_ints_pixels/total_object_pixels;
       per_int_pi2=total_ints_pixels/total_img_pixels;
       
       sum_all1 = sum_all1 + per_int_pi1;
       sum_all2 = sum_all2 + per_int_pi2;
       
       f2=figure('visible','off');
       subplot(1,2,1);imshow(u_mask);
       title(sprintf('PerByObj:%f PerByImg:%f',per_int_pi1,per_int_pi2));
       subplot(1,2,2); imshow(i_mask);
      
       saveas(f2,sprintf('%s_1.jpg',fullfile(opts.voc_imagespath_int,img.file_name(1:end-4))));
       saveas(f1,sprintf('%s_2.jpg',fullfile(opts.voc_imagespath_int,img.file_name(1:end-4))));
       
       close(f2);
       close(f1);
       fprintf(fid,'%s,%s,%s,%s,%s,%s,%s\n',num2str(k),img.file_name(1:end-4),num2str(total_ints_pixels),num2str(total_object_pixels),num2str(total_img_pixels),num2str(per_int_pi1),num2str(per_int_pi2));
       fclose(fid);
 
    
      
end 

       avg_all1= sum_all1/total_imgs ;
       avg_all2= sum_all2/total_imgs ;
 
      fprintf('Occluded By obj%f',avg_all1);
      fprintf('Occluded By image%f',avg_all2);
 
 
 
 

