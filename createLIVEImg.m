clear;clc;close all;

folder = 'E:\Important Works\NRIQA_CNN\LIVE_images\';
names = {'bikes','buildings','buildings2','caps','carnivaldolls','cemetry','churchandcapitol','coinsinfountain',...
    'dancers','flowersonih35','house','lighthouse','lighthouse2','manfishing','monarch','ocean',...
    'paintedhouse','parrots','plane','rapids','sail1','sail2','sail3','sail4','statue','stream','studentsculpture',...
    'woman','womanhat'};

imgsize = 256;
stepsize = imgsize;
downsize = 8;
imgnum = 1000;
ptnum = 6;
ch = 3;
image = zeros(imgsize, imgsize, ch, ptnum, imgnum);
imglabel = zeros(imgnum,1);

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
        in = 0;
        for i = 1:stepsize:(size(img,1)-imgsize+1)
            for j = 1:stepsize:(size(img,2)-imgsize+1)
                in = in + 1;
                idx = i:(i+imgsize-1);
                idy = j:(j+imgsize-1);
                image(:,:,:,in,ind) = img(idx,idy,:);
            end
        end
        tmp = strfind(imname, '-');
        if isempty(tmp)
            imglabel(ind) = 85;
        else
            imglabel(ind) = str2num(imname(tmp(1)+1:tmp(2)-1));
        end
        
    end
end
image = uint8(image);
image = image(:,:,:,:,1:ind);
imglabel = imglabel(1:ind);
imglabel(imglabel>85) = 85;
imglabel = imglabel - min(imglabel);
imglabel = imglabel ./ max(imglabel);
ind = randperm(length(imglabel));
image = image(:,:,:,:,ind);
imglabel = imglabel(ind);

h5create('image_live.h5','/DS1',size(image));
h5create('image_live_lab.h5','/DS1',size(imglabel));
h5write('image_live.h5','/DS1',image);
h5write('image_live_lab.h5','/DS1', imglabel);
disp('finish');