function [ output_args ] = compare_07_12_voc( input_args )
%COMPARE_07_12_VOC Summary of this function goes here
%   Detailed explanation goes here

seg.imgsetpath12='E:\Bharti\Code\Thesis\data\VOC2012\ImageSets\Segmentation\trainval.txt'; 
seg.imgsetpath07='E:\Bharti\Code\Thesis\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\ImageSets\Main\test1.txt';


gtids12=textread(seg.imgsetpath12,'%s');
gtids07=textread(seg.imgsetpath07,'%s');

ids=intersect(gtids12,gtids07);



end

