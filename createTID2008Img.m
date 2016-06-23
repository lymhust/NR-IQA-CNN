clear;clc;close all;

folder = 'E:\Important Works\Image Fingerprint\Test_Images\TID2008_images\';
names = {'beach','bikes','boy','building','caps','door','flower','house',...
    'house2','lighthouse1','lighthouse2','ocean','parrots','plane','rapids','sails1',...
    'sails2','sails3','sails4','statue','stream','wall','woman','womanhat'};

imgsize = 64;
downsize = 8;
imgnum = 40*30;
image = zeros(imgsize, imgsize, imgnum);
bitsize = 64;
cls = 0;
imglabel = zeros(bitsize, imgnum);
imgclass = zeros(imgnum,1);

ind = 0;
lab = 0;

for nm = 1:length(names)
    foldertmp = [folder names{nm} '\'];
    Files = dir(strcat(foldertmp,'*.*'));
    dataNum = length(Files) - 2;
    
    for num_file = 3:length(Files) % Traverse image folder
        ind = ind + 1;
        disp(ind);
        imname = Files(num_file).name;
        img = imread([foldertmp imname]);
        img = rgb2gray(img);
        image(:,:,ind) = imresize(img, [imgsize, imgsize]);
        
        if num_file == 3
             [lab, fin_dct] = pHash_DCT(img);
%             fin_dct = fin_dct - min(fin_dct);
%             lab = fin_dct ./ max(fin_dct);
            
%             img = imresize(img, [downsize, downsize]);
%             lab = double(img(:));
%             lab = lab - min(lab);
%             lab = lab ./ max(lab);
            cls = cls + 1;
        end
        
        imglabel(:,ind) = lab;
        imgclass(ind) = cls;
    end
end
image = uint8(image);
image = image(:,:,1:ind);
imglabel = imglabel(:,1:ind);
imgclass = imgclass(1:ind);

save imgclass_tid imgclass;

h5create('image_tid_64.h5','/DS1',[imgsize, imgsize, ind]);
h5create('image_tid_64_lab.h5','/DS1',[bitsize, ind]);
h5write('image_tid_64.h5','/DS1',image);
h5write('image_tid_64_lab.h5','/DS1', imglabel);
disp('finish');