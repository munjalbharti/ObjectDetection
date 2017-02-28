%% Demo for the CocoApi (see CocoApi.m)
close all;
clear ;

%% initialize COCO api (please specify dataType/annType below)
annTypes = { 'instances'};
dataType='val2014'; annType=annTypes{1}; % specify dataType/annType
annFile=sprintf('../annotations/%s_%s.json',annType,dataType);
coco=CocoApi(annFile);

%% display COCO categories and supercategories
if( ~strcmp(annType,'captions') )
  cats = coco.loadCats(coco.getCatIds());
  nms={cats.name}; fprintf('COCO categories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
  nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
end

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
    'couch'
    'train' 
    'tv' };

catIds = coco.getCatIds('catNms', opts.classes);
imgIds = coco.getImgIds('catIds',catIds);
 
  %'tvmonitor'
  %'sofa' %sofa class not present in coco
  %'motorbike' 
  %'diningtable' 
  %'aeroplane
  %'pottedplant'  
   
   
  opts.constant=2000;
%%{   
opts.nclasses=length(opts.classes);
%opts.voc_imagespath='../all/val_images/';
%opts.voc_labelspath='../all/val_labels/';


%opts.voc_offsetspath='C:\Users\Bharti\Thesis\data\coco\val_offsets_uint8\';
%opts.voc_offsetspath1='C:\Users\Bharti\Thesis\data\coco\train_offsets1_uint8\';
opts.voc_offsetspathmat='C:\Users\Bharti\Thesis\data\coco\val_offsets_mat_int\';
%%}

%{
opts.voc_imagespath='../all/val_images/';
opts.voc_labelspath='../all/val_labels/';
opts.voc_offsetspath='../all/val_offsets/';
opts.voc_offsetspathmat='../all/val_offsets_mat/';
%}


%mkdir(opts.voc_imagespath);
%mkdir(opts.voc_labelspath);
%mkdir(opts.voc_offsetspath);
%mkdir(opts.voc_offsetspathmat);

 max_height=0;
 max_height_img=0;
 max_width=0;
 max_width_img=0;
 
 


tic;
for k=1:length(imgIds)
    %% load and display image
    
     if toc>10
            fprintf('Status %d/%d\n',k,length(imgIds));            
            tic;
     end
        
    imgId = imgIds(k);
    %imgId=254449;
    img = coco.loadImgs(imgId);
  
    if(img.height > max_height)
        max_height=img.height;
        max_height_img=imgId;
    end 
    
    if(img.width > max_width)
        max_width=img.width;
        max_width_img=imgId;
    end 
    
    
    I_orig = imread(sprintf('../images/%s/%s',dataType,img.file_name));
    if(img.height < img.width)
            I=imresize(I_orig,[256,NaN]);
     else 
            I=imresize(I_orig,[NaN,256]);
    end

    
    
    
    img_height=size(I,1);
    img_width=size(I,2);
    
   % f1=figure;  imshow(I); 
   % hold on ;
    %save image

    %% load and display annotations
    annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[0]);
    anns = coco.loadAnns(annIds); 
  %  coco.showAnns(anns);
    
    label_gt=zeros(img_height,img_width,'uint8');
    offset_gt=zeros(img_height,img_width,2,'int16');
    size_gt=zeros(img_height,img_width,2,'uint16');
    
    offset_mask = zeros(img_height,img_width,2,'int16');
    
    all_masks1=coco.getSegmentationMasks(anns,img);
    all_masks=imresize(all_masks1,[img_height,img_width],'nearest');
    i_mask= zeros(img_height,img_width,'logical');
    
    
    total_masks=size(all_masks,3);
    total_ann=size(anns,2);
    if(total_masks ~= total_ann)
        disp('Error');
        break ;
    end 
    
     
     %sort bounding boxes based on area
     [a,sorted_index]=sort([anns.area],'descend');
    % aans=orderfields(anns_o,sorted_index);
    
    for m=sorted_index
        
        bbox=anns(1,m).bbox';
        bbox(1)=bbox(1) * img_width/img.width;
        bbox(3)=bbox(3) * img_width/img.width;
        bbox(2)=bbox(2) * img_height/img.height;
        bbox(4)=bbox(4) * img_height/img.height;
        
        cat_id=anns(1,m).category_id;
        seg_mask=all_masks(:,:,m);
        
        
            ind_all_masks = find(all_masks == cat_id);
            [~,~,z]=ind2sub(size(all_masks),ind_all_masks);
            z=unique(z);
           
            
            all_masks_exc= all_masks(:,:,setdiff([1:total_masks],z));
            if(size(all_masks_exc,3) ~= 0)   
                     u_w_mask= zeros(img_height,img_width,'logical');
                     tmp=sum(all_masks_exc(:,:,:),3);
                     u_w_mask(find(tmp > 0))=1;

                     mask=seg_mask & u_w_mask ;
                     i_mask= i_mask | mask;
            end 
        
        
        
    
        [seg_y,seg_x]=find(seg_mask > 0);
        ind=sub2ind(size(seg_mask),seg_y,seg_x);
        

        c_id=setdiff(unique(seg_mask),0);
        if(c_id ~= cat_id)
            disp('error1');
            break;
        end 
        
      
        
        
        bbx=round(bbox);
        
        %bounding box start from 0
        bbx(1)=bbx(1)+1;
        bbx(2)=bbx(2)+1;
        
        center=[(bbx(2)+floor(bbx(4)/2)),(bbx(1)+floor(bbx(3)/2))];
        center_y = center(1);
        center_x = center(2);
        ct=coco.loadCats(cat_id);
        
        clsind = strmatch(ct.name,opts.classes,'exact');
        
        label_gt(ind)= clsind;
        
        
        for n=1:size(seg_y)
               offset_mask(seg_y(n),seg_x(n),1)  =  center_x-seg_x(n);
               offset_mask(seg_y(n),seg_x(n),2)  =  center_y-seg_y(n);                    
        end 
                
       
        ind1=sub2ind([size(seg_mask),2],seg_y,seg_x,ones(size(seg_y,1),1));
        ind2=sub2ind([size(seg_mask),2],seg_y,seg_x,ones(size(seg_y,1),1)+1);
        size_gt(ind1)=bbx(3);
        size_gt(ind2)=bbx(4);
        
     
    
    end 
    
     inrsctn_mask = ~i_mask;
    
        %{
        o=offset_mask + opts.constant;
        
         if(size(find(o < 0)) > 0)
             fprintf('change constant,current width %d height %d',img.height,img.width );
             return;
         end 
       offset_gt = uint16(o) ;
       %}
     
       offset_gt = offset_mask ;
       
       offset_gt(:,:,1) = offset_gt(:,:,1) .* int16(inrsctn_mask);
       offset_gt(:,:,2) = offset_gt(:,:,2) .* int16(inrsctn_mask);
       size_gt(:,:,1) = size_gt(:,:,1) .* uint16(inrsctn_mask);
       size_gt(:,:,2) = size_gt(:,:,2) .* uint16(inrsctn_mask);
    
     
    
    
      %  f2=figure; 
      %  subplot(2,2,1);
      %  imshow(I); 
         
      %  subplot(2,2,2);
        
        cmap = labelColors(21) ;
        
        %image(label_gt) ;
       % axis image ;
       
        %colormap(cmap) ;  
        
        
       
        label_gt= label_gt .* uint8(inrsctn_mask);
        label_gt =label_gt+1;
        %RGB = ind2rgb(label_gt,cmap);
        %label_gt_r=rgb2ind(RGB,cmap);
        %if(label_gt ~= label_gt_r)
         %   fprintf('ERROR for Image %d',k);
         %   return ;
        %end 
       
     
        
        
      %  imwrite(inrsctn_mask,fullfile(opts.voc_labelspath,sprintf('%s_mask.jpg',img.file_name(1:end-4))));
       
       % subplot(2,2,3);
       % imagesc(offset_gt(:,:,1)-opts.constant);
        
       % subplot(2,2,4);
       % imagesc(offset_gt(:,:,2)-opts.constant);
        
      
         
       
       
        %hold on ;
        %quiver(repmat([1:img.width],img.height,1),repmat([1:img.height]',1,img.width),o(:,:,1)-opts.constant,o(:,:,2)-opts.constant);
     
       if(size(I,3)==1)
               I=repmat(I,[1,1,3]);
       end 
       
       %saviing 16 bit as 8 bits
       size_gt_x_uint8=[uint8(floor(single(size_gt(:,:,1)) ./256)),uint8(rem(size_gt(:,:,1),256))];
       size_gt_y_uint8=[uint8(floor(single(size_gt(:,:,2)) ./256)),uint8(rem(size_gt(:,:,2),256))];
       
       size_gt1_uint8= size_gt;
      % ind=find(size_gt1_uint8 > 256);
      % if(~isempty(ind))
       %   disp('here');
      % end
       size_gt1_uint8(size_gt1_uint8 > 256)=256;
       size_gt1_uint8=size_gt1_uint8-1; %to make it 0 to 255 from 1 to 256
       size_gt1_uint8 =uint8(size_gt1_uint8);
       
       offset_gt_x_uint8= [uint8(floor(single(offset_gt(:,:,1)) ./256)),uint8(rem(offset_gt(:,:,1),256))];
       offset_gt_y_uint8= [uint8(floor(single(offset_gt(:,:,2)) ./256)),uint8(rem(offset_gt(:,:,2),256))];
     
       %imwrite(I,fullfile(opts.voc_imagespath,img.file_name));
       %imwrite(label_gt,fullfile(opts.voc_labelspath,sprintf('%s.png',img.file_name(1:end-4))));
     
      save(fullfile(opts.voc_offsetspathmat,sprintf('%s.mat',img.file_name(1:end-4))), 'offset_gt','size_gt', 'anns', '-v7.3');
       %imwrite(offset_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_ox.png',img.file_name(1:end-4))));
      % imwrite(offset_gt(:,:,2),fullfile(opts.voc_offsetspath,sprintf('%s_oy.png',img.file_name(1:end-4))));
      % imwrite(size_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_sx.png',img.file_name(1:end-4))));
      % imwrite(size_gt(:,:,2),fullfile(opts.voc_offsetspath,sprintf('%s_sy.png',img.file_name(1:end-4))));
       
       
      % imwrite(offset_gt_x_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_ox.png',img.file_name(1:end-4))));
      % imwrite(offset_gt_y_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_oy.png',img.file_name(1:end-4))));
      % imwrite(size_gt_x_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_sx.png',img.file_name(1:end-4))));
      % imwrite(size_gt_y_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_sy.png',img.file_name(1:end-4))));
       
      % imwrite(size_gt1_uint8(:,:,1),fullfile(opts.voc_offsetspath1,sprintf('%s_sx.png',img.file_name(1:end-4))));
      % imwrite(size_gt1_uint8(:,:,2),fullfile(opts.voc_offsetspath1,sprintf('%s_sy.png',img.file_name(1:end-4))));
       
     
      %}

         
       
        
        
       % close(f1);
       % close(f2);
        
        
        
          
        
    %save label
    %save offset
    %save size
  
    
    
end 

 fprintf('Max Height %d Image %d\n',max_height,max_height_img);  
 fprintf('Max Width %d Image %d\n',max_width,max_width_img); 
%imgId = imgIds(randi(length(imgIds)));


