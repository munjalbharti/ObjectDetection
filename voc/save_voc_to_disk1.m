clear ;
close all;
VOCopts.seg.imgsetpath='E:\Bharti\Code\Thesis\data\VOC2012\ImageSets\Segmentation\val.txt';
%VOCopts.trainset='train';
VOCopts.annopath='E:\Bharti\Code\Thesis\data\VOC2012\Annotations\';
VOCopts.imgpath='E:\Bharti\Code\Thesis\data\VOC2012\JPEGImages\';


%VOCinit;
%opts.voc_imagespath='C:\Users\Bharti\Thesis\data\voc\train_images\';
%opts.voc_labelspath='C:\Users\Bharti\Thesis\data\voc\val_labels\';
%opts.voc_offsetspath='C:\Users\Bharti\Thesis\data\voc\val_offsets_uint8\';
%opts.voc_offsetspath1='C:\Users\Bharti\Thesis\data\voc\train_offsets1_uint8\';
opts.voc_offsetspathmat='C:\Users\Bharti\Thesis\data\voc\val_offsets_mat_int\';

%mkdir(opts.voc_imagespath);
%mkdir(opts.voc_labelspath);
%mkdir(opts.voc_offsetspath);
%mkdir(opts.voc_offsetspath1);
mkdir(opts.voc_offsetspathmat);


opts.constant=2000;

seg_class_dir='E:\Bharti\Code\Thesis\data\VOC2012\SegmentationClass\';
seg_obj_dir='E:\Bharti\Code\Thesis\data\VOC2012\SegmentationObject\';
gtids=textread(VOCopts.seg.imgsetpath,'%s');

for i=1:length(gtids)
 
 
  recs=PASreadrecord([VOCopts.annopath,sprintf('%s.xml',gtids{i})]);
  
   
   I_orig = imread([VOCopts.imgpath,sprintf('%s.jpg',gtids{i})]);
    if(size(I_orig,1) < size(I_orig,2))
            I=imresize(I_orig,[256,NaN]);
     else 
            I=imresize(I_orig,[NaN,256]);
    end
    
    img_height=size(I,1);
    img_width=size(I,2);
    
    label_gt=zeros(img_height,img_width,'uint8');
    offset_gt=zeros(img_height,img_width,2,'int16');
    size_gt=zeros(img_height,img_width,2,'uint16');
   
    offset_mask = zeros(img_height,img_width,2,'int16');
 
   
    [seg_mask,~]=imread(fullfile(seg_class_dir,sprintf('%s.png',gtids{i})));  
    [seg_obj,~]=imread(fullfile(seg_obj_dir,sprintf('%s.png',gtids{i})));  
    
    
     seg_mask=imresize(seg_mask,[img_height,img_width],'nearest');
     seg_obj=imresize(seg_obj,[img_height,img_width],'nearest');
  
     seg_mask(seg_mask == 255)=0;
     seg_obj(seg_obj==255)=0;

     
     %inrsctn_mask= false(size(seg_mask),'logical');
     %inrsctn_mask(seg_mask ~=0 & seg_mask ~= 255)=1;
     %in voc, bounding box starts from (1,1) and are given as [xmin,ymin,xmax,ymax]
     label_gt = seg_mask;
     uniq_vals=setdiff(unique(seg_obj),[0]);
      
    % no_of_objects = size(recs.objects,2);
   

      
     for m=1:length(uniq_vals)
               [seg_y,seg_x]=find(seg_obj == uniq_vals(m));
            
                x_min=min(seg_x); x_max=max(seg_x);               
                y_min=min(seg_y); y_max=max(seg_y);                
              
                bbx_width =  x_max-x_min+1;
                bbx_height = y_max-y_min+1;
                
                
                center_x  = x_min+floor(bbx_width/2);
                center_y =  y_min+floor(bbx_height/2);
       

      %{
        for m=1:no_of_objects
        
           bbox=recs.objects(1,m).bbox';
           bbox(1)=bbox(1) * img_width/size(I_orig,2);
           bbox(3)=bbox(3) * img_width/size(I_orig,2);
           bbox(2)=bbox(2) * img_height/size(I_orig,1);
           bbox(4)=bbox(4) * img_height/size(I_orig,1);
           
      
        
           bbx=round(bbox);
        
           %bounding box start from 0
           %bbx(1)=bbx(1)+1;
           %bbx(2)=bbx(2)+1;
        
           center=[bbx(2) + floor((bbx(4)-bbx(2)+1)/2),bbx(1)+ floor((bbx(3)- bbx(1)+1)/2)];
           center_y = center(1);
           center_x = center(2);
           
           bbx_height=bbx(4)-bbx(2)+1;
           bbx_width= bbx(3)-bbx(1)+1;
           [seg_y,seg_x]=find(seg_obj == m);
         %}  
          
           
       
           for n=1:size(seg_y)
               offset_mask(seg_y(n),seg_x(n),1)  =  center_x-seg_x(n);
               offset_mask(seg_y(n),seg_x(n),2)  =  center_y-seg_y(n);                    
           end 
                
       
            ind1=sub2ind([size(seg_obj),2],seg_y,seg_x,ones(size(seg_y,1),1));
            ind2=sub2ind([size(seg_obj),2],seg_y,seg_x,ones(size(seg_y,1),1)+1);
            size_gt(ind1)=bbx_width;
            size_gt(ind2)=bbx_height;       
                
     end 
       %{
        o=offset_mask + opts.constant;
        
        if(size(find(o < 0)) > 0)
             fprintf('change constant,current width %d height %d',img.height,img.width );
             return;
        end 
        offset_gt = uint16(o) ;
        %}
     
        offset_gt=offset_mask;
        %offset_gt(:,:,1) = offset_gt(:,:,1) .* uint16(inrsctn_mask);
        %offset_gt(:,:,2) = offset_gt(:,:,2) .* uint16(inrsctn_mask);
        %size_gt(:,:,1) = size_gt(:,:,1) .* uint16(inrsctn_mask);
        %size_gt(:,:,2) = size_gt(:,:,2) .* uint16(inrsctn_mask);
        %label_gt = label_gt .* uint8(inrsctn_mask);
        
        label_gt= label_gt+1;
        cmap = labelColors(21) ;
        
        
       size_gt_x_uint8=[uint8(floor(single(size_gt(:,:,1)) ./256)),uint8(rem(size_gt(:,:,1),256))];
       size_gt_y_uint8=[uint8(floor(single(size_gt(:,:,2)) ./256)),uint8(rem(size_gt(:,:,2),256))];
       
       size_gt1_uint8= size_gt;
      % ind=find(size_gt1_uint8 > 255);
      % if(~isempty(ind))
      %    disp('here');
      % end
       size_gt1_uint8(size_gt1_uint8 > 256)=256;
       size_gt1_uint8=size_gt1_uint8-1;
       size_gt1_uint8 =uint8(size_gt1_uint8);
       
       offset_gt_x_uint8= [uint8(floor(single(offset_gt(:,:,1)) ./256)),uint8(rem(offset_gt(:,:,1),256))];
       offset_gt_y_uint8= [uint8(floor(single(offset_gt(:,:,2)) ./256)),uint8(rem(offset_gt(:,:,2),256))];
     
         
     %  imwrite(I,fullfile(opts.voc_imagespath,sprintf('%s.jpg',gtids{i})));
     %  imwrite(label_gt,fullfile(opts.voc_labelspath,sprintf('%s.png',gtids{i})));
  
       save(fullfile(opts.voc_offsetspathmat,sprintf('%s.mat',gtids{i})), 'offset_gt','size_gt', '-v7.3');
      %  imwrite(offset_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_ox.png',gtids{i})));
      %  imwrite(offset_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_oy.png',gtids{i})));
      %  imwrite(size_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_sx.png',gtids{i})));
      %  imwrite(size_gt(:,:,2),fullfile(opts.voc_offsetspath,sprintf('%s_sy.png',gtids{i})));
       
     
    %imwrite(offset_gt_x_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_ox.png',gtids{i})));
    %imwrite(offset_gt_y_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_oy.png',gtids{i})));
    %imwrite(size_gt_x_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_sx.png',gtids{i})));
    %imwrite(size_gt_y_uint8,fullfile(opts.voc_offsetspath,sprintf('%s_sy.png',gtids{i})));
    %imwrite(size_gt1_uint8(:,:,1),fullfile(opts.voc_offsetspath1,sprintf('%s_sx.png',gtids{i})));
    %imwrite(size_gt1_uint8(:,:,2),fullfile(opts.voc_offsetspath1,sprintf('%s_sy.png',gtids{i})));
       
      
         
       


end 
