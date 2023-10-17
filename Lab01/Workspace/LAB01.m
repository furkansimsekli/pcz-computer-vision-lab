% EXERCISE 1
% 
% img1 = imread('lenna.png');
% imshow(img1);
% 
% img2 = rgb2gray(img1);
% figure;
% imshow(img2);
% imwrite(img2, 'lennaSz.png');
%
% EXERCISE 2
% img1 = imread('lenna.png');
% img2 = imread('daenerys.png');
% subplot(2, 2, 1);
% imshow(img1);
% subplot(2, 2, 2);
% imshow(img2);
% subplot(2, 2, 3);
% imshowpair(img1, img2, 'montage');
%
% EXERCISE 3
% cd Pom1
% gg = dir('*.png');
% size = length(gg);
% cd ../
% 
% for i=1:size
%     cd Pom1
%     img = imread(gg(i).name);
%     figure;
%     imshow(img)
%     grayImg = rgb2gray(img);
%     figure;
%     imshow(grayImg);
%     cd ../
%     cd Pom2
%     imwrite(grayImg, gg(i).name);
%     cd ../
% end
%
% EXERCISE 4
cd Pom1
gg = dir('*.png');
size = length(gg);
cd ../

for i=1:size
    cd Pom1
    img = imread(gg(i).name);
    figure;
    imshow(img);
    grayImg = rgb2gray(img);
    resizedImg = imresize(grayImg, [227, 227]);
    figure;
    imshow(resizedImg);
    cd ../
    cd Pom3
    imwrite(resizedImg, gg(i).name);
    cd ../
end
