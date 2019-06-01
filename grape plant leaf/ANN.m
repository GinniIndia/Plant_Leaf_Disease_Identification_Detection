%  Project Title    : Plant Disease Classification
%  Diseases Analyzed: Grape Disease Detection
%  Author Name      : Ginni Garg, Mantosh Biswas
%  E mail I.D       : gargginni01@gmail.com
%------------------------------------------------------------------------------------------------------------------
% This script assumes these variables are defined:
%
%   grape_input - input data.
%   grape_output - target data.
load input_grape;
load output_grape;

x = input_grape';
t = output_grape';
trainFcn = 'trainbr';

% Create a Pattern Recognition Network
hiddenLayerSize = 1;
net = patternnet(hiddenLayerSize, trainFcn);
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%net.trainParam.showWindow = false;
% Train the Network
[net,tr] = train(net,x,t);
%------------------------------------------------------
clc
% Select an image from the 'Disease Dataset' folder by opening the folder
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick a Disease Affected Leaf');
he = imread([pathname,filename]);


    imshow(he), title('H&E image');
    text(size(he,2),size(he,1)+15,...
     'Image courtesy of Alan Partin, Johns Hopkins University', ...
     'FontSize',7,'HorizontalAlignment','right');
    cform = makecform('srgb2lab');
    lab_he = applycform(he,cform);
    ab = double(lab_he(:,:,2:3));
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,2);

   
    %------------------------------------------------------------------------
    %Automatic Selection of No. of cluster (Elbow Method)
    dim=size(ab);
    % default number of test to get minimun under differnent random centriods
    test_num=10;
    distortion=zeros(floor(dim(1)/5000),1);
    for k_temp=1:floor(dim(1)/5000)
    [~,~,sumd]=kmeans(ab,k_temp,'emptyaction','drop');
    destortion_temp=sum(sumd);
    % try differnet tests to find minimun disortion under k_temp clusters
    for test_count=2:test_num
        [~,~,sumd]=kmeans(ab,k_temp,'emptyaction','drop');
        destortion_temp=min(destortion_temp,sum(sumd));
    end
    distortion(k_temp,1)=destortion_temp;   
    end

    variance=distortion(1:end-1)-distortion(2:end);
    distortion_percent=cumsum(variance)/(distortion(1)-distortion(end));
    %plot(distortion_percent,'b*--');

    [r,~]=find(distortion_percent>0.9);
    K=r(1,1)+1

    %------------------------------------------------------------------------------------
     %Used for Healthy Leaf
    if K<=2
        disp("Healthy Leaf");
        return 
    end
    nColors = K;
    % repeat the clustering 3 times to avoid local minima
    [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
                                  
    pixel_labels = reshape(cluster_idx,nrows,ncols);
    imshow(pixel_labels,[]), title('image labeled by cluster index');
    segmented_images = cell(1,3);
    rgb_label = repmat(pixel_labels,[1 1 3]);

    for k = 1:nColors
        color = he;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
    end


% Display the contents of the clusters
if nColors>=1
figure, subplot(nColors,2,1);imshow(segmented_images{1});title('Cluster 1'); 
end
if nColors>=2
subplot(nColors,2,2);imshow(segmented_images{2});title('Cluster 2');
end
if nColors>=3
subplot(nColors,2,3);imshow(segmented_images{3});title('Cluster 3');
end
if nColors>=4
subplot(nColors,2,4);imshow(segmented_images{4});title('Cluster 4');
end
if nColors>=5
subplot(nColors,2,5);imshow(segmented_images{5});title('Cluster 5');
end
if nColors>=6
subplot(nColors,2,6);imshow(segmented_images{6});title('Cluster 6');
end
% Feature Extraction

% The input dialogue makes sure that we extract features only from the
% disease affected part of the leaf
x = inputdlg('Enter the cluster no. containing the disease affected leaf part only:');
i = str2double(x);

seg_img = segmented_images{i};

% Convert to grayscale if image is RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end

% Create the Gray Level Cooccurance Matrices (GLCMs)
glcms = graycomatrix(img);

%Evaluate 13 features from the disease affected region only
% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
% Put the 13 features in an array
input_image = [Contrast,Correlation,Energy,Homogeneity,Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
input_image=input_image';
y = net(input_image);
[val,idx]=max(y);
if idx==1
    disp("Disease:Black_Rot")
else
    disp("Disease:Black_Measles(ESCA)")
end

