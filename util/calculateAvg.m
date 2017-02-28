filename='E:\Bharti\Code\Thesis\data\rgb_object_detection-offset-all-class-L-0.001-normalised\Results1\VOCval\486\hough_th_500_bin_16_win_5\ALL_0_perct\results.csv';
fid=fopen(filename,'r');
C = textscan(fid,'%s,%s,%s %s %s %s','Delimiter',{',','/'});
ap1=C{2,:};
avg=sum(ap1(:))/20;

 %fclose(fileID);
%[class,ap,other]=textread(filename,'%s,%f,%s');

   