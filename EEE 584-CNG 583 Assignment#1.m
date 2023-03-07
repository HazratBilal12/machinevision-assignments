clear all;
clc;

% EEE 584 Assignment # 1     Hazrat Bilal(2542611) 

%%  Question 1: Read Frame1 and Convert to Gray-Scale Image 

Frame1 = imread('D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame1.jpg');
figure;
imshow(Frame1)
title('Frame1 RGB Image');

Frame1_gs = rgb2gray(Frame1);                   % Frame1 Gray Scale Image
figure;
imshow(Frame1_gs);
title('Frame1 Gray Scale Image');

%% Question 2: Find Gradient using the ‘Sobel Operator’ and Show the Edge Strength Map

[Gx,Gy] = imgradientxy(Frame1_gs);           % Sobel is the default Method
figure;
imshowpair(Gx, Gy, 'montage');
title('Gray Scale Image Gradients in x-direction(Gx) and y-direction(Gy) using Sobel Operator ');


Frame1_gs_Edge = edge(Frame1_gs, 'Canny');   % Sobel is the default Method
figure;
imshow(Frame1_gs_Edge);
title('Edge Strenght of Gray Scale Image of Frame1');


%% Question 3: Generate and show the intensity histogram of gray-scale Frame1

figure;
imhist(Frame1_gs);
xlabel('Number of Gray Levels');
ylabel('Number of Pixels');
title('Histogram of Gray Scale Frame1 Image ');

%% Question 4: Generate intensity histograms of gray-scale Frame1, Frame20 and Frame35. Then, compute
%              the Manhattan distance Which of the two images are the most similar?


Frame20 = imread('D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame20.jpg');
Frame35 = imread('D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame35.jpg');

Frame1_gs = Frame1_gs;
Frame20_gs = rgb2gray(Frame20);
Frame35_gs = rgb2gray(Frame35);
% figure; imshow(Frame1_gs);
% figure; imshow(Frame20_gs);
% figure; imshow(Frame35_gs);

Hist_Frame1 = imhist(Frame1_gs);

figure; 
stem(Hist_Frame1);
title('Histogram of Frame 1')

Hist_Frame20 = imhist(Frame20_gs);

figure; 
stem(Hist_Frame20);
title('Histogram of Frame 20')

Hist_Frame35 = imhist(Frame35_gs);

figure; 
stem(Hist_Frame35);
title('Histogram of Frame 35')

D_1_20=sum((abs(Hist_Frame1-Hist_Frame20)))
D_1_35=sum((abs(Hist_Frame1-Hist_Frame35)))
D_20_35=sum((abs(Hist_Frame20-Hist_Frame35)))

%% Question 5: Write a Code for Background Modeling with Univariate Gaussian Density Function.
% Display both the Mean and Standard Deviation Images.

% Mean for the Given 39 Frames

Frame1 = Frame1;

Sum_Frames = double(rgb2gray(Frame1));

for i=2:39
    Frames = imread(['D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame',num2str(i),'.jpg']);
    Sum_Frames = Sum_Frames + double(rgb2gray(Frames));
end

mean_Frames = Sum_Frames/39;
figure(); 
imshow(mean_Frames, []);
title("Mean of Frames for Background Detection");

% Standard Deviation for the Given Frames

Squre_Frames = double(0);
for i=1:39
    RGB_Frame = imread(['D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame',num2str(i),'.jpg']);
    Gray_Frame = double(rgb2gray(RGB_Frame));
    Squre_Frames = Squre_Frames + (Gray_Frame - mean_Frames).^2; 
end
    
 STD_Frames = sqrt((Squre_Frames)/38);
 
 figure();
 imshow(STD_Frames, []); 
 title("Standard Deviation of Given Frames for Background");

% Background of the Given 39 Frames

 Background_Frames = mean_Frames + STD_Frames;
 figure(); 
 imshow(Background_Frames, []); 
 title("Background of Given Frames by combining Mean & Standared Deviation");
 
 
 % For "Frame30.jpg" generate alikelihood value for each pixel.
 
RGB_Frame30 = imread('D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame30.jpg');
Frame30 = double(rgb2gray(RGB_Frame30));

PDF_Frame30 = normpdf(Frame30, mean_Frames, STD_Frames);
 
% figure(); 
% imshow(PDF_Frame30, []); 
% title("Combined images ");
 
Bin_Seg_Img = double( PDF_Frame30 < 0.0006);
 
 %meanMimage10 = meanMimage10 > 0.008;
 %figure(), imshow(image10b .* meanMimage10, []);
 
figure(); 
imshow(Bin_Seg_Img); 
title("Binary Segmented Image with Threshold(0.006)");
 
 
imageminusmean = Frame30 - mean_Frames;
figure(); 
imshow(imageminusmean, []); 
title("difference of Frame30 and Mean of All Frames");

%% Question 6: Use a Ready Code of Lucas-Kanade Optical Flow Algorithm.

Frames_Folder ='D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder';
Frames_n = dir(fullfile(Frames_Folder, '*.jpg'));
total_frames=39;
 
for i= 1:total_frames
    f= fullfile(Frames_Folder, Frames_n(i).name);
    Framesf6{i} = rgb2gray(imread(f)) ;
end  
  
 
Frame3=3;                                                        
Frame5=5;
 
fig = figure;
hViewPanel = uipanel(fig,'Position',[0 0 1 1], 'Title','Plot of Optical Flow Vectors');
Optflow_Plot = axes(hViewPanel);

opticFlow = opticalFlowLK('NoiseThreshold', 0.0006); 


for  i=Frame3:Frame5
     flow = estimateFlow(opticFlow,double(Framesf6{i}));
end

imshow('D:\Hazrat Bilal\Fall 2021-22\Machine Vision 584\HW1\Frames Folder\Frame5.jpg');
hold on
plot(flow,'DecimationFactor',[5 5],'ScaleFactor',5,'Parent',Optflow_Plot);
hold off

Opt_flow_Img = flow.Magnitude;

figure(); 
imshow(Opt_flow_Img);
title('Optical Flow Magnitude Image');

PDF_Opt_flow_Img = normpdf(Opt_flow_Img);



Bin_Seg_with_Threshold = double(PDF_Opt_flow_Img < 0.008);

figure();
imshow(Bin_Seg_with_Threshold);
title('Binary Segmented Image')





