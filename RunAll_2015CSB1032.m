function [] = RunAll_2015CSB1032()

	% Loading training data
	train_images = loadMNISTImages('train-images-idx3-ubyte');
	train_labels = loadMNISTLabels('train-labels-idx1-ubyte');

	% The dimensions of the training data
	no_images = size(train_images, 2);
	dimension_images = sqrt(size(train_images, 1));

	%% Loading/Creating features

	% Filename containing the features of the images
	feature_filename = 'features.csv';

	% Parameters for the global descriptor
	block_size = 8;
	no_blocks = 36; %There will be 50% overlap between the blocks
	gradient_bins = 9;

	if ( exist(feature_filename, 'file') == 0 )
	    disp('Creating features');
	    all_image_features = zeros(no_images, (no_blocks*gradient_bins)+1);
	    for i = 1:no_images
	        if (rem(i,1000)==1)
	            fprintf('%dth image\n', i);
	        end
	        ith_image = reshape(train_images(:,i), 28, 28);
	        for j = 0:no_blocks-1
	            row = floor(j/sqrt(no_blocks));
	            col = rem(j,sqrt(no_blocks));
	            patch_image = ith_image(row*4+1:row*4+block_size, col*4+1:col*4+block_size);
	            patch_descriptor = ComputePatchDescriptor_2015CSB1032(patch_image, gradient_bins);
	            all_image_features(i, j*gradient_bins+1:((j+1)*gradient_bins)) = patch_descriptor;
	        end
	        all_image_features(i, no_blocks*gradient_bins+1) = train_labels(i);
	    end
	    csvwrite(feature_filename,all_image_features);
	else
	    disp('Loading features');
	    all_image_features = csvread(feature_filename);
	end

	%% Loading/Creating dictionary

	% Filename containing the dictionary of words
	dictionary_filename = 'dictionary.csv';

	% Parameters for the dictionary
	no_words = 100;

	if ( exist(dictionary_filename, 'file') == 0 )
	    disp('Creating dictionary');
	    [dictionary, words_index] = CreateDictionary_2015CSB1032(all_image_features, no_words);
            disp(words_index);
            for i=1:size(words_index, 1)
               row =  train_images(:, words_index(i));
               word_image = reshape(row, 28, 28);
               imwrite(word_image, strcat('words/',num2str(i),'.jpg'));
            end
        csvwrite(dictionary_filename, dictionary);
	else
	    disp('Loading dictionary');
	    dictionary = csvread(dictionary_filename);
	end


	%% Loading/Computing histograms for all images

	% Filename containing the histogram of train_images
	train_histogram_filename = 'train_histogram.csv';

	if ( exist(train_histogram_filename, 'file') == 0 )
	    disp('Computing training histograms');
	    train_histogram = zeros(no_images, no_words);
	    for i=1:no_images
	        if (rem(i,1000)==1)
	            fprintf('%dth image\n', i);
	        end
	        one_histogram = ComputeHistogram_2015CSB1032(all_image_features(i,:), dictionary);
	        train_histogram(i, :) = one_histogram;
	    end
	    csvwrite(train_histogram_filename, train_histogram);
	else
	    disp('Loading training histograms');
	    train_histogram = csvread(train_histogram_filename);
	end

	% Loading testing data
	test_images = loadMNISTImages('t10k-images-idx3-ubyte');
	test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
	no_test_images = size(test_images, 2);

	% Filename containing the features of the images
	test_feature_filename = 'test_features.csv';

	if ( exist(test_feature_filename, 'file') == 0 )
	    disp('Creating test features');
	    test_image_features = zeros(no_test_images, (no_blocks*gradient_bins)+1);
	    for i = 1:no_test_images
	        if (rem(i,1000)==1)
	            fprintf('%dth image\n', i);
	        end
	        ith_image = reshape(test_images(:,i), 28, 28);
	        for j = 0:no_blocks-1
	            row = floor(j/sqrt(no_blocks));
	            col = rem(j,sqrt(no_blocks));
	            patch_image = ith_image(row*4+1:row*4+block_size, col*4+1:col*4+block_size);
	            patch_descriptor = ComputePatchDescriptor_2015CSB1032(patch_image, gradient_bins);
	            test_image_features(i, j*gradient_bins+1:((j+1)*gradient_bins)) = patch_descriptor;
	        end
	        test_image_features(i, no_blocks*gradient_bins+1) = test_labels(i);
	    end
	    csvwrite(test_feature_filename, test_image_features);
	else
	    disp('Loading features');
	    test_image_features = csvread(test_feature_filename);
	end


	% Filename containing the histogram of test images
	test_histogram_filename = 'test_histogram.csv';

	if ( exist(test_histogram_filename, 'file') == 0 )
	    disp('Computing test histograms');
	    test_histogram = zeros(no_test_images, no_words);
	    for i=1:no_test_images
	        if (rem(i,1000)==1)
	            fprintf('%dth image\n', i);
	        end
	        one_histogram = ComputeHistogram_2015CSB1032(test_image_features(i,:), dictionary);
	        test_histogram(i, :) = one_histogram;
	    end
	    csvwrite(test_histogram_filename, test_histogram);
	else
	    disp('Loading testing histograms');
	    test_histogram = csvread(test_histogram_filename);
	end

	%% Histogram matching

	% Now, we have train histogram (train_histogram), and test histogram (test_histogram)
	% For each image of test histogram, we will calculate its distance
	% with all the train histograms, the label of the closest distance train
	% histogram will be the predicted label for that test image.
	% Then, we can calculate the accuracy measure by comparison with true label

	disp('Calculating test accuracy');
	disp('It takes around 13 minutes. Be patient, have coffee');

	test_total = size(test_histogram, 1);
	train_total = size(train_histogram, 1);
	accuracy = 0;
	confusion_matrix = zeros(10, 10);

	predicted_labels = zeros(test_total,1);

	parfor i = 1:test_total
	    histogram1 = test_histogram(i,:);
	    histogram1 = meshgrid(histogram1, 1:train_total);
	    distances = MatchHistogram_2015CSB1032(histogram1, train_histogram);
	    
	    k = 50;
	    mink = [];
	    for x=1:k
	        [~, indexx] = min(distances);
	        distances(indexx,1) = 100000;
	        mink = [mink, train_labels(indexx)];
	    end
	        
	    predicted_labels(i,1) = mode(mink);
	        
	end

	for i=1:test_total
	    if ( predicted_labels(i,1) == test_labels(i,1) )
	        accuracy = accuracy+1;
	    end
	    confusion_matrix(test_labels(i,1)+1, predicted_labels(i,1)+1) = confusion_matrix(test_labels(i,1)+1, predicted_labels(i,1)+1)+1;    
	end

	classwise_accuracy = zeros(10,1);
	classwise_precision = zeros(10,1);
	classwise_recall = zeros(10,1);

	for i=1:10
	   tp = confusion_matrix(i,i);
	   fp = sum(confusion_matrix(:,i))-confusion_matrix(i,i);
	   fn = sum(confusion_matrix(i,:))-confusion_matrix(i,i);
	   N = sum(sum(confusion_matrix));
	   tn =  N-(tp+fp+fn);
	   classwise_accuracy(i,1) = ((tp+tn)/N)*100;
	   classwise_precision(i,1) = (tp/(tp+fp))*100;
	   classwise_recall(i,1) = (tp/(tp+fn))*100;
	end

	disp('The confusion matrix : ');
	disp(confusion_matrix);
	disp('The classwise accuracy (ith index corresponds to class i): ');
	disp(classwise_accuracy);
	disp('The classwise precision (ith index corresponds to class i): ');
	disp(classwise_precision);
	disp('The classwise recall (ith index corresponds to class i): ');
	disp(classwise_recall);

	fprintf('Overall accuracy = %f\n', (accuracy/N)*100);

end

function [ distance ] = MatchHistogram_2015CSB1032(histogram1, histogram2)
    no_histogram = size(histogram1, 1);
    bins = size(histogram1, 2);
        
    diff = histogram1-histogram2;
    diff = diff.^2;
    add = histogram1+histogram2;
    
    hist = diff./add;
    distance = sum(hist, 2);
    
end

function [ dictionary, words_index ] = CreateDictionary_2015CSB1032( features, no_words )

    no_features = size(features, 1);
    feature_dimension = size(features, 2) - 1;
    ClusterCenters = zeros(no_words, feature_dimension);
    
    ClusterCenters = datasample(features(:, 1:feature_dimension), no_words);
    
    threshold = 1.5;
    change = 10;
    
    iter = 1;
    disp('k-means iterations:')
    while (change > threshold)
        disp(iter);
        % disp(change);
        iter = iter+1;
        ClusterSum = zeros(no_words, feature_dimension);
        ClusterNo = zeros(no_words, 1);
        
        % Assign each feature a cluster center
        for i = 1:no_features
            feature = features(i, 1:feature_dimension);
            distance_from_centers = sum( ((ClusterCenters-meshgrid(feature,1:no_words)).^2)' );
            [~,chosen_center] = min(distance_from_centers);
                        
            ClusterSum(chosen_center, :) = ClusterSum(chosen_center, :) + feature;
            ClusterNo(chosen_center, 1) = ClusterNo(chosen_center, 1) + 1;
        end
        
        old_centers = ClusterCenters;
        
        for i = 1:no_words
            if (ClusterNo(i,1) ~= 0)
                ClusterCenters(i,:) = ClusterSum(i,:)/ClusterNo(i,1);
            end
        end
        
        change = sqrt(sum(sum((ClusterCenters-old_centers).^2)));
    end
    
    % Assigning label to each cluster center
    % For each feature, we will vote its label to the closest center
    % For each center, the label with highest votes would be the final
    % label
    cluster_label_count = zeros(no_words, 10);
    
    % Assign each feature a cluster center, then increase the vote of the
    % feature label for that cluster center
    for i = 1:no_features
        feature = features(i, 1:feature_dimension);
        distance_from_centers = sum( ((ClusterCenters-meshgrid(feature,1:no_words)).^2)' );
        [~,chosen_center] = min(distance_from_centers);
        
        feature_label = features(i, feature_dimension+1)+1;
        cluster_label_count(chosen_center, feature_label) = cluster_label_count(chosen_center, feature_label) + 1;
    end
    
    words_index = zeros(no_words, 1);
    for i=1:no_words
        histogram1 = meshgrid(ClusterCenters(i,:), 1:no_features);
        histogram2 = features(:, 1:feature_dimension);        
        diff = histogram1-histogram2;
        diff = diff.^2;
        add = histogram1+histogram2;
        hist = diff./add;
        distance = sum(hist, 2);
        [~, ithindex] = min(distance);
        words_index(i, 1) = ithindex;
    end
    
    [~, cluster_labels] = max(cluster_label_count');
    dictionary = [ClusterCenters (cluster_labels'-1)];
    
end



function [ descriptor ] = ComputePatchDescriptor_2015CSB1032( image, gradient_bins )
    
    shape = size(image);
    descriptor = zeros(1, gradient_bins);
    [Gmag, Gdir] = imgradient(image);
    Gdir = abs(Gdir);
    
    for i = 1:shape(1)
        for j= 1:shape(2)
            weight2 = rem(Gdir(i,j), 20)/20;
            weight1 = 1-weight2;
            bin_no = ceil(Gdir(i,j)/20);
            if (bin_no == 0)
                bin_no = 1;
            end
            bin_no2 = bin_no+1;
            if (bin_no2 == 10)
                bin_no2 = 1;
            end
            descriptor(1, bin_no) = descriptor(1, bin_no) + (weight1*Gmag(i,j));
            descriptor(1, bin_no2) = descriptor(1, bin_no2) + (weight2*Gmag(i,j));
        end
    end
    
end


function [ histogram ] = ComputeHistogram_2015CSB1032( feature, dictionary )
    
    feature_dimension = size(feature, 2)-1;
    no_words = size(dictionary, 1);
    histogram = zeros(1, no_words);
    dict = dictionary(:, 1:feature_dimension);
    f = meshgrid(feature(1, 1:feature_dimension),1:no_words);

    distance_from_centers = sum( ((dict-f).^2 )' );
    histogram = (distance_from_centers.^(-1)).^2;
    
    histogram = histogram ./ sum(histogram);
        
end




function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end


function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end


	