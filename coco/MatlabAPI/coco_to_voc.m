%% Demo for the CocoApi (see CocoApi.m)
close all;
clear ;

%% initialize COCO api (please specify dataType/annType below)
annTypes = { 'instances'};
dataType='train2014'; annType=annTypes{1}; % specify dataType/annType
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
    'sofa'
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
   
   
  opts.constant=5000;
 %{   
opts.nclasses=length(opts.classes);
opts.voc_imagespath='../val_images/';
opts.voc_labelspath='../val_labels/';
opts.voc_offsetspath='../val_offsets/';
%}

%%{
opts.voc_imagespath='../train_images/';
opts.voc_labelspath='../train_labels/';
opts.voc_offsetspath='../train_offsets/';
%%}


mkdir(opts.voc_imagespath);
mkdir(opts.voc_labelspath);
mkdir(opts.voc_offsetspath);

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
    img = coco.loadImgs(imgId);
    
    if(img.height > max_height)
        max_height=img.height;
        max_height_img=imgId;
    end 
    
    if(img.width > max_width)
        max_width=img.width;
        max_width_img=imgId;
    end 
    
    
    I = imread(sprintf('../images/%s/%s',dataType,img.file_name));
    %f1=figure;  imshow(I); 
    %hold on ;
    %save image

    %% load and display annotations
    annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[0]);
    anns = coco.loadAnns(annIds); 
    %coco.showAnns(anns);
    
    label_gt=zeros(img.height,img.width,'uint8');
    offset_gt=zeros(img.height,img.width,2,'uint16');
 
    size_gt=zeros(img.height,img.width,2,'uint16');
    size_gt(:)=NaN;
    
    offset_mask = zeros(img.height,img.width,2);
    offset_mask(:)=NaN;
     
     %sort bounding boxes based on area
     [a,sorted_index]=sort([anns.area],'descend');
    % aans=orderfields(anns_o,sorted_index);
    
    for m=sorted_index
        
        bbox=anns(1,m).bbox';
        cat_id=anns(1,m).category_id;

        
        %bounding box start from 0
        bbx=round(bbox)+1;
        center=[round(bbx(2)+bbx(4)/2),round(bbx(1)+bbx(3)/2)];
        ct=coco.loadCats(cat_id);
        
        clsind = strmatch(ct.name,opts.classes,'exact');
        
        label_gt([bbx(2):bbx(2)+bbx(4)-1],[bbx(1):bbx(1)+bbx(3)-1])= clsind;
        
              
        offset_mask([bbx(2):bbx(2)+bbx(4)-1],[bbx(1):bbx(1)+bbx(3)-1],1)= repmat(center(2)-[bbx(1):bbx(1)+bbx(3)-1],bbx(4),1) ;
        offset_mask([bbx(2):bbx(2)+bbx(4)-1],[bbx(1):bbx(1)+bbx(3)-1],2)= repmat([center(1)-[bbx(2):bbx(2)+bbx(4)-1]]',1,bbx(3)) ;
              
       
      
       % rectangle('Position',[bbx(1),bbx(2),bbx(3),bbox(4)],'linewidth',3);
       % plot(center(2),center(1),'r+');
     
       
       
       
        
        size_gt([bbx(2):bbx(2)+bbx(4)-1],[bbx(1):bbx(1)+bbx(3)-1],1)= img.width;
        size_gt([bbx(2):bbx(2)+bbx(4)-1],[bbx(1):bbx(1)+bbx(3)-1],2)= img.height;
       
       
     
    
    end 
    
         o=offset_mask + opts.constant;
        
         if(size(find(o < 0)) > 0)
             fprintf('change constant,current width %d height %d',img.height,img.width );
             return;
         end 
         %saving as uint16
        offset_gt = o;
        
       % f2=figure; 
        %subplot(2,2,1);
        %imshow(I); 
         
        %subplot(2,2,2);
        
        cmap = labelColors(21) ;
        
        %image(label_gt) ;
        %axis image ;
       
        colormap(cmap) ;
               
         
        RGB = ind2rgb(label_gt,cmap);
       % imshow(RGB);
        label_gt_r=rgb2ind(RGB,cmap);
        if(label_gt ~= label_gt_r)
            fprintf('ERROR for Image %d',k);
            return ;
        end 
        % tmp2=getframe;
      
        imwrite(RGB,fullfile(opts.voc_labelspath,img.file_name));
       
        %subplot(2,2,3);
        %imagesc(offset_gt(:,:,1)-opts.constant);
        
        %subplot(2,2,4);
        %imagesc(offset_gt(:,:,2)-opts.constant);
        
      
         
       
       
        %hold on ;
        %quiver(repmat([1:img.width],img.height,1),repmat([1:img.height]',1,img.width),o(:,:,1)-opts.constant,o(:,:,2)-opts.constant);
     
        
     
       imwrite(I,fullfile(opts.voc_imagespath,img.file_name));
    
       save(fullfile(opts.voc_offsetspath,sprintf('%s.mat',img.file_name(1:end-4))), 'offset_gt','size_gt', 'anns', '-v7.3');
       
     
       
         
       
        
        
       % close(f1);
       % close(f2);
        
        
        
          
        
    %save label
    %save offset
    %save size
  
    
    
end 

 fprintf('Max Height %d Image %d\n',max_height,max_height_img);  
 fprintf('Max Width %d Image %d\n',max_width,max_width_img); 
%imgId = imgIds(randi(length(imgIds)));


