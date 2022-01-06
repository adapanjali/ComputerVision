% CE4003 Computer Vision
% Lab 01: 
% Point Processing + Spatial Filtering + Frequency Filtering + Imaging Geometry

%% 1 Objectives 

% This laboratory aims to introduce image processing in MATLAB context. 
% In this laboratory you will:
% a. Become familiar with the MATLAB and Image Processing Toolbox software package.
% b. Experiment with the point processing operations of contrast stretching and histogram equalization.
% c. Evaluate how different Gaussian and median filters are suitable for noise removal.
% d. Become familiar with the frequency domain operations
% e. Understand imaging geometry.

%% 2.1 Contrast Stretching 

% a1 Reading the image as MATLAB matrix 
Pc = imread('mrt-train.jpeg');
whos Pc

% a2 Converting the image into grey-scale image 
P = rgb2gray(Pc);
whos P

% b Viewing the image 
figure; imshow(P); title("Original Image")

% c Checking the minimum and maximum intensities present in the image
min(P(:)) 
max(P(:))

% d1 Peforming contrast stretching
P2 = 255/191*imsubtract(P,13);

% d2 Checking the minimum and maximum intensities again 
min(P2(:)) % The minimum intensity is 0
max(P2(:)) % The maximum intensity is 255

% e Displaying the image again
figure; imshow(P2); title('Contrast Stretching')

%% 2.2 Histogram Equalisation 

% a1 Displaying the image intensity histogram of P using 10 bins
figure; imhist(P, 10); title("Histogram with 10 bins (Original)")

% a2 Displaying the image intensity histogram of P using 256 bins
figure; imhist(P, 256); title("Histogram with 256 bins (Original)")

% QUESTION 
% What are the differences?

% b1 Performing histogram equalisation 
P3 = histeq(P, 255);

% b2 Redisplaying the histograms for P3 with 10 bins
figure; imhist(P3, 10); title("Histogram with 10 bins(Histogram Equilization)")

% b3 Redisplaying the histograms for P3 with 256 bins
figure; imhist(P3, 256); title("Histogram with 256 bins (Histogram Equilization)")

% b4 Histogram equalised image
figure; colormap('gray'); imshow(uint8(P3)); title("Histogram Equilised Image")

% QUESTION
% Are the histograms equalized? 
% What are the similarities and differences between the latter two histograms?

% c Rerunning the histogram equalization on P3
 P4 = histeq(P3, 255);
  
 figure; imhist(P4, 10); title("Histogram with 10 bins(Histogram Equilization Rerun)")
 figure; imhist(P4, 256); title("Histogram with 256 bins (Histogram Equilization Rerun)")

 % QUESTION 
 % Does the histogram become more uniform? 
 % Give suggestions as to why this occurs.

 %% 2.3 Linear Spatial Filtering

 % a1 Generating Gaussian filters 
 G =@ (X, Y, Sigma) exp(-(X.^2 + Y.^2)./(Sigma^2.*2))./(2*pi*Sigma^2);


 dim = 5
 x = -(dim - 1)/2 : (dim - 1)/2;
 y = -(dim - 1)/2 : (dim - 1)/2;

 [X, Y] = meshgrid(x, y);

 % a2 Gaussian filter with sigma = 1.0
 G1 = G(X, Y, 1)
 G1 = G1/sum(G1(:))

 % a3 Gaussian filter with sigma = 2.0
 G2 = G(X, Y, 2)
 G2 = G2/sum(G2(:))

 figure; mesh(G1);
 figure; mesh(G2);

 % b Reading the image as MATLAB matrix
 P = imread('ntu-gn.jpeg');
 whos P
 figure; imshow(P); title("Original Image")

 % c Applying the Gaussian averaging filters on the above image
 P1d = double(P);
 P1dc = conv2(G1, P);
 P1 = uint8(P1dc);
 figure; imshow(P1);

 P2d = double(P);
 P2dc = conv2(G2, P);
 P2 = uint8(P2dc);
 figure; imshow(P2);

 % QUESTION 
 % How effective are the filters in removing noise? 
 % What are the trade-offs between using either of the two filters, 
 % or not filtering the image at all?

 % d Reading the image as MATLAB matrix
 P = imread('ntu-sp.jpeg');
 whos P
 figure; imshow(P);

 % e Applying the Gaussian averaging filters on the above image
 P1d = double(P);
 P1dc = conv2(G1, P);
 P1 = uint8(P1dc);
 figure; imshow(P1);

 P2d = double(P);
 P2dc = conv2(G2, P);
 P2 = uint8(P2dc);
 figure; imshow(P2);

 % QUESTION 
 % Are the filters better at handling Gaussian noise or speckle noise?

 %% 2.4 Median Filtering 

 % a Reading the image as MATLAB matrix
 P = imread('ntu-gn.jpeg');
 whos P
 figure; imshow(P);

 % b1 Applying the 3x3 Median filter on the above image
 P1d = double(P);
 P1dc = medfilt2(P1d, [3 3]);
 P1 = uint8(P1dc);
 figure; imshow(P1);

 % b1 Applying the 5x5 Median filter on the above image
 P2d = double(P);
 P2dc = medfilt2(P2d, [5 5]);
 P2 = uint8(P2dc);
 figure; imshow(P2);

 % d Reading the image as MATLAB matrix
 P = imread('ntu-sp.jpeg');
 whos P
 figure; imshow(P);

 % e Applying the Gaussian averaging filters on the above image
 P1d = double(P);
 P1dc = medfilt2(P1d, [3 3]);
 P1 = uint8(P1dc);
 figure; imshow(P1);

 P2d = double(P);
 P2dc = medfilt2(P2d, [5 5]);
 P2 = uint8(P2dc);
 figure; imshow(P2);

 % QUESTION
 % How does Gaussian filtering compare with median filtering in 
 % handling the different types of noise? 
 % What are the tradeoffs?

 %% 2.5 Suppressing Noise Interference Patterns: Bandpass Filtering 

 % a Reading the image as MATLAB matrix
 P = imread('pck-int.jpeg');
 whos P 
 figure; imshow(P); title("Original Image")

 % b1 Obtaining the Fourier ransform F of the image using fft2
 F = fft2(P);

 % b2 Computing the power spectrum S
 S = abs(F);

 % b3 Displaying the power spectrum with fftshift
 figure; imagesc(fftshift(S.^0.1)); colormap('default'); title("fft shift")
 figure; imagesc(fftshift(S.^0.1)); colormap('gray'); title("fft shift (Grey)")
 %[x, y] = ginput(2)

 % c Redisplay the power spectrum without fftshift
 figure; imagesc(S.^0.1); colormap('default'); title("fft")
 figure; imagesc(S.^0.1); colormap('gray'); title("fft (Grey)")
 [x, y] = ginput(2)

 % x = [9.6429 248.5369]
 % y = [241.1997 16.5466]

 % d1 Set to zero the 5x5 neighbourhood elements at locations corresponding 
 % to the above peaks in the Fourier transform F, not the power spectrum
 x1 = 9;
 y1 = 241;
 x2 = 249;
 y2 = 17;
 m = 2;
 
 F(y2 - m : y2 + m, x2 - m : x2 + m) = zeros(2*m + 1);
 F(y1 - m : y1 + m, x1 - m : x1 + m) = zeros(2*m + 1);

 % d2 Recomputing the power spectrum and displaying it
 S = abs(F);
 
 % d3 Displaying the power spectrum
 figure; imagesc(fftshift(S.^0.1)); colormap('default'); title("fft shift")
 figure; imagesc(fftshift(S.^0.1)); colormap('gray'); title("fft shift (Grey)")

 figure; imagesc(S.^0.1); colormap('default'); title("fft")
 figure; imagesc(S.^0.1); colormap('gray'); title("fft (Grey)")

 % e Computing the inverse Fourier transform using ifft2 and displaying the resultant image
 figure; colormap('gray'); imshow(uint8(ifft2(F)))

 % QUESTION
 % Comment on the result and how this relates to step (c). 
 % Can you suggest any way to improve this?

 %After setting zero the 5x5 neighborhood elements, 
 % there are actually still some white line on the spectrum. 
 % To improve the image, we can set those white part to zero

 [h,w] = size(F);
 F(:,x1) = zeros(h,1);
 F(:,x2) = zeros(h,1);
 F(y1,:) = zeros(1,w);
 F(y2,:) = zeros(1,w);

 S = abs(F);

 figure; imagesc(S.^0.1); colormap('default');

%% 2.5 f

% Reading the image 
Pc = imread('primate-caged.jpeg');

%Check for RGB or grayscale image
whos P

% Change to grayscale
P = rgb2gray(Pc);
figure; imshow(P); title("Original Image")

% Fourier Transform 2D
fftp = fft2(P);

% Check for spectrum, we can see there are 4 tiny spectrum causes the effect
figure; imagesc(fftshift(real(fftp.^0.5))); 
figure; imagesc(fftshift(real(fftp.^0.5))); colormap('gray');

fftp_zero = fftp;
figure; imagesc(real(fftp_zero.^0.5));
figure; imagesc(real(fftp_zero.^0.5)); colormap('gray');

x1=251; y1=11;  fftp_zero(x1-3:x1+3,y1-3:y1+3) = 0;
x2=10;  y2=237; fftp_zero(x2-3:x2+3,y2-3:y2+3) = 0;
x3=247; y3=21;  fftp_zero(x3-3:x3+3,y3-3:y3+3) = 0;
x4=6;   y4=247; fftp_zero(x4-3:x4+3,y4-3:y4+3) = 0;

figure; imagesc(real(fftp_zero.^0.5));
figure; imagesc(real(fftp_zero.^0.5)); colormap('gray');

%Perform Inverse Fourier Transform
filted_fftp_zero = uint8(ifft2(fftp_zero));
figure; imshow(real(filted_fftp_zero))

%% 2.6 Undoing Perspective Distortion of Planar Surface

% a Reading the image
  P=imread('book.jpeg');
  whos P

  figure; imshow(P); title("Original Image")

 % b Getting the coordinates of the four corners of the image 
 % [X Y] = ginput(4);

  x = [143; 309; 4; 257];
  y = [28; 47; 158; 215];
  
  xn = [0; 210; 0; 210];
  yn = [0; 0; 297; 297];

  % c Setting up the matrices required to estimate the projective transformation
  A = [
       x(1), y(1), 1, 0, 0, 0, -xn(1)*x(1), -xn(1)*y(1);
       0, 0, 0, x(1), y(1), 1, -yn(1)*x(1), -yn(1)*y(1);
       x(2), y(2), 1, 0, 0, 0, -xn(2)*x(2), -xn(2)*y(2);
       0, 0, 0, x(2), y(2), 1, -yn(2)*x(2), -yn(2)*y(2);
       x(3), y(3), 1, 0, 0, 0, -xn(3)*x(3), -xn(3)*y(3);
       0, 0, 0, x(3), y(3), 1, -yn(3)*x(3), -yn(3)*y(3);
       x(4), y(4), 1, 0, 0, 0, -xn(4)*x(4), -xn(4)*y(4);
       0, 0, 0, x(4), y(4), 1, -yn(4)*x(4), -yn(4)*y(4); ]

  v = [xn(1); yn(1); xn(2); yn(2); xn(3); yn(3); xn(4); yn(4)]

  u = A \ v;

  U = reshape([u;1], 3, 3)';

  w = U*[X'; Y'; ones(1,4)]; 
  w = w ./ (ones(3,1) * w(3,:))

  % QUESTION 
  % Does the transformation give you back the 4 corners of the desired image?

  % d Warping the image
  T = maketform('projective', U');
  P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

  % e Displaying the image 
  figure; imshow(P2)

  % QUESTION 
  % Is this what you expect? 
  % Comment on the quality of the transformation and suggest reasons.




