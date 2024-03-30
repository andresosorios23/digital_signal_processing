clc;
clear;

%% Leer las 3 imagenes:
Lim1=imread('assets/im1.jpg');%%% se leen imagenes y se transforman en matrices matrices %
Lim2=imread('assets/im2.jpg');
Lim3=imread('assets/im3.jpg');

%% Se convierte a blanco y negro para poder realizar la correlacion en 2D:
Im1gr=rgb2gray(Lim1);
Im2gr=rgb2gray(Lim2);
Im3gr=rgb2gray(Lim3);
[y1,x1]=size(Im1gr);
[y2,x2]=size(Im1gr);
[y3,x3]=size(Im1gr);
%figure(1);imshow(Im1gr);title('Imagen original 1');
%figure(2);imshow(Im2gr);title('Imagen original 2');
%figure(3);imshow(Im3gr);title('Imagen original 3');
%% Se recortan las imagenes 1 y 2 para calcular en esa area la correlaci�n:
sub1 = imcrop(Im1gr,[x1/2 0 x1/2 y1]);
sub2 = imcrop(Im2gr,[0 0 x2/5 y2]);

%% correlacion de la imagen 1 y 2:
corrpro1=xcorr2(sub1,sub2);
%% Desplazamiento encontrado por la correlaci�n entre la imagen 1 y la imagen 2:
[maxcorr,loc] = max(abs(corrpro1(:)));
[ypeak, xpeak] = ind2sub(size(corrpro1),loc(1));
corr_offset = [(size(sub1,1)-xpeak)
    (ypeak-size(sub1,2))];
xoffset=corr_offset(1);

%% Nueva imagen (entre imagen 1 y 2):
sub3 = imcrop(Im1gr,[0 1 xoffset 394]);
tt2=[sub3,Im2gr];
[yn,xn]=size(tt2);
%figure(4)
%imshow(tt2); title('Nueva imagen entre 1 y 2 en escala de grises');

%% Se recortan las imagenes 3 y la imagen nueva (1 y 2):
sub2_2 = imcrop(tt2,[xn/6 0 xn yn]);
[yn1,xn1] = size(sub2_2);
sub4 = imcrop(Im3gr,[0 0 x3/2 y3]);

%% correlacion de la imagen 3 y la nueva:
corrpro2=xcorr2(sub4,sub2_2);

%% Desplazamiento encontrado por la correlaci�n entre la imagen nueva y la 3:
[maxcorr2,loc2] = max(abs(corrpro2(:)));
[ypeak2, xpeak2] = ind2sub(size(corrpro2),loc2(1));
corr_offset2 = [(size(sub4,1)-xpeak2)
    (ypeak2-size(sub4,2))];
xoffset2=corr_offset2(1);
yoffset2=corr_offset2(2);

%% Nueva imagen (entre imagen nueva y 2):
sub6 = imcrop(Im3gr,[x3-xoffset2-xpeak+9 0 x3 y3]);
tt3=[tt2,sub6];
figure(10)
imshow(tt3);title('Imagen panoramica completa en escala de grises');

%% Procedimiento en color:
% Sacando elemento de la imagen 1 a partir de la correlaci�n y uniendolo con la imagen 2:
tt4=imcrop(Lim1,[0 1 xoffset y1]);
tt5=[tt4,Lim2];
% Sacando elemento de la imagen 3 a partir de la correlaci�n y uniendolo con la union de la imagen 1 y 2:
tt6 = imcrop(Lim3,[x3-xoffset2-xpeak+9 0 x3 y3]);
tt7=[tt5,tt6];
figure(11)
imshow(tt7);title('Imagen panoramica completa a color');

