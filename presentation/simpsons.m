clc; clear all, close all
%%
im1= imread('c.jpg');
im1g = rgb2gray(im1);
figure;imshow(im1g);title('Imagen original');
xi=150;
xf=325;
yi=475;
yf=640;
im2g=im1g(xi:xf,yi:yf);
figure;imshow(im2g)
%%
nim1g = im1g-mean(mean(im1g));
figure;imshow(nim1g);
nim2g = nim1g(xi:xf,yi:yf);
%corr = xcorr2(nim1g,nim2g);
corr=xcorr2(im1g,im2g);
[maxcorr,loc] = max(corr(:));
[X,Y] = ind2sub(size(corr),loc);
%%
figure; plot(corr(:))
title('Correlación cruzada')
hold on
plot(loc,maxcorr,'or')
hold off
text(loc*1.05,maxcorr,'Máximo')
%%
figure; imagesc(im1g)
axis image off
colormap gray
title('Localización encontrada')
hold on
plot([Y-size(im2g,2) Y-size(im2g,2) Y Y Y-size(im2g,2)],[X-size(im2g,1) X X X-size(im2g,1) X-size(im2g,1)],'r')
hold off
