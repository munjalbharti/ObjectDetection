%SETUP Setting up project paths
[pathstr,name,ext] = fileparts(mfilename('fullpath')); 
% 
addpath([pathstr filesep]);
addpath([pathstr filesep 'util']);
addpath([pathstr filesep 'util' filesep 'data_preparation']);
addpath([pathstr filesep 'net']);
addpath([pathstr filesep 'data']);
addpath([pathstr filesep 'PostProcess']);
addpath([pathstr filesep 'Evaluation']);

if (ispc)
    %window specific stuff here
    extern_folder='E:\Bharti\Code\Thesis\';
    addpath(extern_folder);
    
    addpath([extern_folder filesep 'ResNet']);
    addpath([extern_folder filesep 'extern']);
    

    matconvnet_folder=[extern_folder filesep 'extern\matconvnet-1.0-beta23-win\'];
    addpath(matconvnet_folder);  
    
    
    
    
    
else 
     matconvnet_folder='matconvnet-1.0-beta23';
end

addpath(matconvnet_folder);
addpath([matconvnet_folder filesep 'matlab' filesep]);


%folders are inside project folder
%addpath([pathstr filesep 'extern' filesep matconvnet_folder filesep]);
%addpath([pathstr filesep 'extern' filesep matconvnet_folder filesep 'matlab' filesep]);


%addpath([pathstr filesep 'extern' filesep 'vlfeat' filesep 'toolbox' filesep]);


vl_setupnn
%vl_setup
