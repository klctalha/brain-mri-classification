% Adım 1: Veriseti Hazırlama (Train / Validation / Test Ayrımı)
projectRoot = pwd;

datasetPath = fullfile(projectRoot, 'archive', 'brain_tumor_dataset');
outputBasePath = fullfile(projectRoot, 'archive', 'brain_tumor_dataset_resized');
saveFolder = fullfile(projectRoot, 'trainNetworkProject');

if ~exist(datasetPath, 'dir')
    error(['Hata: Veri seti bulunamadı. Lütfen "archive" klasörünü ' ...
           'proje ana dizinine yerleştirin. Aranan yol: %s'], datasetPath);
end
if ~exist(outputBasePath, 'dir')
    mkdir(outputBasePath);
end


datasetPath = fullfile(projectRoot, 'archive', 'brain_tumor_dataset');

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imds = shuffle(imds);

[imdsTrain, imdsRest] = splitEachLabel(imds, 0.7, 'randomized'); 
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized'); 
outputBasePath = fullfile(projectRoot, 'archive', 'brain_tumor_dataset_resized');


mkdir(fullfile(outputBasePath, 'train', 'yes'));
mkdir(fullfile(outputBasePath, 'train', 'no'));
mkdir(fullfile(outputBasePath, 'validation', 'yes'));
mkdir(fullfile(outputBasePath, 'validation', 'no'));
mkdir(fullfile(outputBasePath, 'test', 'yes'));
mkdir(fullfile(outputBasePath, 'test', 'no'));

inputSize = [224 224];


for i = 1:length(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    label = imdsTrain.Labels(i);
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, inputSize);
    filename = sprintf('train_%d.jpg', i);
    imwrite(img, fullfile(outputBasePath, 'train', char(label), filename));
end

for i = 1:length(imdsValidation.Files)
    img = readimage(imdsValidation, i);
    label = imdsValidation.Labels(i);
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, inputSize);
    filename = sprintf('val_%d.jpg', i);
    imwrite(img, fullfile(outputBasePath, 'validation', char(label), filename));
end

for i = 1:length(imdsTest.Files)
    img = readimage(imdsTest, i);
    label = imdsTest.Labels(i);
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, inputSize);
    filename = sprintf('test_%d.jpg', i);
    imwrite(img, fullfile(outputBasePath, 'test', char(label), filename));
end


% Adım 2: Görüntüleri Eğitime Hazırlama (Boyutlandırma ve Augmentasyon)

trainPath = fullfile(outputBasePath, 'train');
validationPath = fullfile(outputBasePath, 'validation');
testPath = fullfile(outputBasePath, 'test');

imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsValidation = imageDatastore(validationPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

inputSize = [224 224];

augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ... 
    'RandXTranslation', [-5 5], ... 
    'RandYTranslation', [-5 5], ... 
    'RandXScale', [0.9 1.1], ... 
    'RandYScale', [0.9 1.1]);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);

augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);


% Adım 3: Kendi CNN Modelimizi Tanımlama

layers = [
    imageInputLayer([224 224 3], 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')

    fullyConnectedLayer(128, 'Name', 'fc_1')
    reluLayer('Name', 'relu_4')
    dropoutLayer(0.5, 'Name', 'dropout')

    fullyConnectedLayer(2, 'Name', 'fc_2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% analyzeNetwork(layers);

%{
%% Adım 4.1: CNN - Adam Optimizasyonu ile Eğitim ve Hiperparametre Kaydı

optionsAdam = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 5);

[net_cnn_adam, info_adam] = trainNetwork(augimdsTrain, layers, optionsAdam);

classNames = categories(imdsTrain.Labels);

netCustom = net_cnn_adam;
save('cnn_adam_best_model.mat', 'netCustom', 'classNames', '-v7.3');
disp('En iyi CNN Adam modeli başarıyla kaydedildi.');

hyperParams = table;
hyperParams.LossFunction = "categorical crossentropy";
hyperParams.NumEpochs = optionsAdam.MaxEpochs;
hyperParams.BatchSize = optionsAdam.MiniBatchSize;
hyperParams.OptimizationAlgorithm = "adam";
hyperParams.LearningRate = optionsAdam.InitialLearnRate;
hyperParams.ActivationFunction = "softmax";
hyperParams.NumDenseUnits = 128;
hyperParams.ValidationFraction = 0.2; 

writetable(hyperParams, 'cnn_adam_hyperparameters.csv');
disp('En iyi CNN Adam hiperparametre tablosu kaydedildi.');

%}

%{
%% Adım 4.2: CNN - Adam Eğitilmiş Model ile Test ve Yüzdelik Performans

load('cnn_adam_best_model.mat', 'netCustom', 'classNames');

predictedLabels_adam = classify(netCustom, augimdsTest);
trueLabels = imdsTest.Labels;

confMat_adam = confusionmat(trueLabels, predictedLabels_adam);

TP = confMat_adam(2,2);
FP = confMat_adam(1,2);
FN = confMat_adam(2,1);
TN = confMat_adam(1,1);

accuracy_adam = (TP + TN) / (TP + TN + FP + FN);
precision_adam = TP / (TP + FP);
sensitivity_adam = TP / (TP + FN);
f1_score_adam = 2 * (precision_adam * sensitivity_adam) / (precision_adam + sensitivity_adam);

fprintf('CNN Adam Test Performansı (%% olarak):\n');
fprintf('Accuracy: %.2f%%\n', accuracy_adam * 100);
fprintf('Precision: %.2f%%\n', precision_adam * 100);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity_adam * 100);
fprintf('F1-Score: %.2f%%\n', f1_score_adam * 100);

figure;
confusionchart(trueLabels, predictedLabels_adam);
title('Confusion Matrix - CNN Adam (Test)');

%}

%{
%% Adım 4.3: CNN - RMSprop Optimizasyonu ile Eğitim ve Hiperparametre Kaydı (Learning Rate Sabit)

optionsRMSprop = trainingOptions('rmsprop', ...
    'InitialLearnRate', 1e-4, ...        
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 5);

[net_cnn_rmsprop, info_rmsprop] = trainNetwork(augimdsTrain, layers, optionsRMSprop);

classNames = categories(imdsTrain.Labels);

netCustom = net_cnn_rmsprop;
save('cnn_rmsprop_best_model.mat', 'netCustom', 'classNames', '-v7.3');
disp('En iyi CNN RMSprop modeli başarıyla kaydedildi.');

hyperParams = table;
hyperParams.LossFunction = "categorical crossentropy";
hyperParams.NumEpochs = optionsRMSprop.MaxEpochs;
hyperParams.BatchSize = optionsRMSprop.MiniBatchSize;
hyperParams.OptimizationAlgorithm = "rmsprop";
hyperParams.LearningRate = optionsRMSprop.InitialLearnRate;
hyperParams.ActivationFunction = "softmax";
hyperParams.NumDenseUnits = 128;
hyperParams.ValidationFraction = 0.2;

writetable(hyperParams, 'cnn_rmsprop_hyperparameters.csv');
disp('En iyi CNN RMSprop hiperparametre tablosu kaydedildi.');

%}

%{
%% Adım 4.4: CNN - RMSprop Eğitilmiş Model ile Test ve Yüzdelik Performans

load('cnn_rmsprop_best_model.mat', 'netCustom', 'classNames');

predictedLabels_rmsprop = classify(netCustom, augimdsTest);
trueLabels = imdsTest.Labels;

confMat_rmsprop = confusionmat(trueLabels, predictedLabels_rmsprop);

TP = confMat_rmsprop(2,2);
FP = confMat_rmsprop(1,2);
FN = confMat_rmsprop(2,1);
TN = confMat_rmsprop(1,1);

accuracy_rmsprop = (TP + TN) / (TP + TN + FP + FN);
precision_rmsprop = TP / (TP + FP);
sensitivity_rmsprop = TP / (TP + FN);
f1_score_rmsprop = 2 * (precision_rmsprop * sensitivity_rmsprop) / (precision_rmsprop + sensitivity_rmsprop);

fprintf('CNN RMSprop Test Performansı (%% olarak):\n');
fprintf('Accuracy: %.2f%%\n', accuracy_rmsprop * 100);
fprintf('Precision: %.2f%%\n', precision_rmsprop * 100);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity_rmsprop * 100);
fprintf('F1-Score: %.2f%%\n', f1_score_rmsprop * 100);

figure;
confusionchart(trueLabels, predictedLabels_rmsprop);
title('Confusion Matrix - CNN RMSprop (Test)');

%}

%{
%% Adım 4.5: CNN - SGDM Optimizasyonu ile Eğitim ve Hiperparametre Kaydı

optionsSGDM = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-4, ...
    'Momentum', 0.9, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 5);

[net_cnn_sgdm, info_sgdm] = trainNetwork(augimdsTrain, layers, optionsSGDM);

classNames = categories(imdsTrain.Labels);

netCustom = net_cnn_sgdm;
save('cnn_sgdm_best_model.mat', 'netCustom', 'classNames', '-v7.3');
disp('CNN SGDM modeli başarıyla kaydedildi.');

hyperParams = table;
hyperParams.LossFunction = "categorical crossentropy";
hyperParams.NumEpochs = optionsSGDM.MaxEpochs;
hyperParams.BatchSize = optionsSGDM.MiniBatchSize;
hyperParams.OptimizationAlgorithm = "sgdm";
hyperParams.LearningRate = optionsSGDM.InitialLearnRate;
hyperParams.ActivationFunction = "softmax";
hyperParams.NumDenseUnits = 128;
hyperParams.ValidationFraction = 0.2;

writetable(hyperParams, 'cnn_sgdm_hyperparameters.csv');
disp('CNN SGDM hiperparametre tablosu kaydedildi.');

%}

%{
%% Adım 4.6: CNN - SGDM Eğitilmiş Model ile Test ve Yüzdelik Performans

load('cnn_sgdm_best_model.mat', 'netCustom', 'classNames');

predictedLabels_sgdm = classify(netCustom, augimdsTest);
trueLabels = imdsTest.Labels;

confMat_sgdm = confusionmat(trueLabels, predictedLabels_sgdm);

TP = confMat_sgdm(2,2);
FP = confMat_sgdm(1,2);
FN = confMat_sgdm(2,1);
TN = confMat_sgdm(1,1);

accuracy_sgdm = (TP + TN) / (TP + TN + FP + FN);
precision_sgdm = TP / (TP + FP);
sensitivity_sgdm = TP / (TP + FN);
f1_score_sgdm = 2 * (precision_sgdm * sensitivity_sgdm) / (precision_sgdm + sensitivity_sgdm);

fprintf('CNN SGDM Test Performansı (%% olarak):\n');
fprintf('Accuracy: %.2f%%\n', accuracy_sgdm * 100);
fprintf('Precision: %.2f%%\n', precision_sgdm * 100);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity_sgdm * 100);
fprintf('F1-Score: %.2f%%\n', f1_score_sgdm * 100);

figure;
confusionchart(trueLabels, predictedLabels_sgdm);
title('Confusion Matrix - CNN SGDM (Test)');

%}

%{

%% Eğitim, Doğrulama, Test ve Toplam Veri Dağılımı Tablosu ve Grafikleri

trainCounts = countcats(imdsTrain.Labels);

validationCounts = countcats(imdsValidation.Labels);

testCounts = countcats(imdsTest.Labels);

allLabels = [imdsTrain.Labels; imdsValidation.Labels; imdsTest.Labels];
totalCounts = countcats(allLabels);

classNames = categories(imdsTrain.Labels);

distributionTable = table( ...
    trainCounts, validationCounts, testCounts, totalCounts, ...
    'RowNames', classNames, ...
    'VariableNames', {'Egitim', 'Dogrulama', 'Test', 'Toplam'});

disp('Veri Dağılımı Tablosu:');
disp(distributionTable);

%% Eğitim Verisi İçin Sınıf Dağılımı Grafiği

figure;
bar(categorical(classNames), trainCounts);
ylabel('Görüntü Sayısı');
xlabel('Sınıf');
title('Eğitim Verisi Sınıf Dağılımı');
grid on;

for i = 1:length(trainCounts)
    text(i, trainCounts(i)+2, num2str(trainCounts(i)), 'HorizontalAlignment', 'center');
end

%% Doğrulama Verisi İçin Sınıf Dağılımı Grafiği

figure;
bar(categorical(classNames), validationCounts);
ylabel('Görüntü Sayısı');
xlabel('Sınıf');
title('Doğrulama Verisi Sınıf Dağılımı');
grid on;

for i = 1:length(validationCounts)
    text(i, validationCounts(i)+2, num2str(validationCounts(i)), 'HorizontalAlignment', 'center');
end

%% Test Verisi İçin Sınıf Dağılımı Grafiği

figure;
bar(categorical(classNames), testCounts);
ylabel('Görüntü Sayısı');
xlabel('Sınıf');
title('Test Verisi Sınıf Dağılımı');
grid on;

for i = 1:length(testCounts)
    text(i, testCounts(i)+2, num2str(testCounts(i)), 'HorizontalAlignment', 'center');
end

%% Adam, RMSprop ve SGDM Test Doğruluklarının Karşılaştırılması
accuracy_adam = 0.9595;     
accuracy_rmsprop = 0.9324;  
accuracy_sgdm = 0.9459;     

accuracies = [accuracy_adam, accuracy_rmsprop, accuracy_sgdm] * 100;
optimizers = categorical({'Adam', 'RMSprop', 'SGDM'});

figure;
bar(optimizers, accuracies);
ylabel('Test Accuracy (%)');
xlabel('Optimizer');
ylim([0 100]);
title('Adam - RMSprop - SGDM Test Accuracy Karşılaştırması');
grid on;

for i = 1:length(accuracies)
    text(i, accuracies(i)+1, sprintf('%.2f%%', accuracies(i)), 'HorizontalAlignment', 'center');
end

%}

%{
%% Eğitim, Doğrulama, Test ve Toplam Veri Dağılımı Tablosu ve Grafikleri

trainCounts = countcats(imdsTrain.Labels);

validationCounts = countcats(imdsValidation.Labels);

testCounts = countcats(imdsTest.Labels);

allLabels = [imdsTrain.Labels; imdsValidation.Labels; imdsTest.Labels];
totalCounts = countcats(allLabels);

classNames = categories(imdsTrain.Labels);

distributionTable = table( ...
    trainCounts, validationCounts, testCounts, totalCounts, ...
    'RowNames', classNames, ...
    'VariableNames', {'Egitim', 'Dogrulama', 'Test', 'Toplam'});

disp('Veri Dağılımı Tablosu:');
disp(distributionTable);

%% Eğitim Verisi İçin Sınıf Dağılımı Grafiği

figure;
bar(categorical(classNames), trainCounts);
ylabel('Görüntü Sayısı');
xlabel('Sınıf');
title('Eğitim Verisi Sınıf Dağılımı');
grid on;

for i = 1:length(trainCounts)
    text(i, trainCounts(i)+2, num2str(trainCounts(i)), 'HorizontalAlignment', 'center');
end

savefig(fullfile(saveFolder, 'egitim_verisi_dagilimi.fig'));

%% Doğrulama Verisi İçin Sınıf Dağılımı Grafiği

figure;
bar(categorical(classNames), validationCounts);
ylabel('Görüntü Sayısı');
xlabel('Sınıf');
title('Doğrulama Verisi Sınıf Dağılımı');
grid on;

for i = 1:length(validationCounts)
    text(i, validationCounts(i)+2, num2str(validationCounts(i)), 'HorizontalAlignment', 'center');
end

savefig(fullfile(saveFolder, 'dogrulama_verisi_dagilimi.fig'));

%% Test Verisi İçin Sınıf Dağılımı Grafiği

figure;
bar(categorical(classNames), testCounts);
ylabel('Görüntü Sayısı');
xlabel('Sınıf');
title('Test Verisi Sınıf Dağılımı');
grid on;

for i = 1:length(testCounts)
    text(i, testCounts(i)+2, num2str(testCounts(i)), 'HorizontalAlignment', 'center');
end

savefig(fullfile(saveFolder, 'test_verisi_dagilimi.fig'));

%% Adam, RMSprop ve SGDM Test Doğruluklarının Karşılaştırılması

accuracy_adam = 0.9595;     
accuracy_rmsprop = 0.9324;  
accuracy_sgdm = 0.9459;    

accuracies = [accuracy_adam, accuracy_rmsprop, accuracy_sgdm] * 100;
optimizers = categorical({'Adam', 'RMSprop', 'SGDM'});

figure;
bar(optimizers, accuracies);
ylabel('Test Accuracy (%)');
xlabel('Optimizer');
ylim([0 100]);
title('Adam - RMSprop - SGDM Test Accuracy Karşılaştırması');
grid on;

for i = 1:length(accuracies)
    text(i, accuracies(i)+1, sprintf('%.2f%%', accuracies(i)), 'HorizontalAlignment', 'center');
end

savefig(fullfile(saveFolder, 'optimizer_karsilastirma_accuracy.fig'));

%}

%{
%% Adım 6.1: MobileNetV2 - Adam Optimizasyonu ile Eğitim ve Hiperparametre Kaydı

net = mobilenetv2;

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels)); % 2 sınıf: normal ve hastalıklı

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_output')
];

% (MobilenetV2'nin son katmanlarının doğru isimleri şunlardır:)
lgraph = removeLayers(lgraph, {'Logits', 'Logits_softmax', 'ClassificationLayer_Logits'});

lgraph = addLayers(lgraph, newLayers);

lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

optionsAdam = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 5);

augimdsTrainTransfer = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augimdsValidationTransfer = augmentedImageDatastore(inputSize(1:2), imdsValidation);

[net_mobilenetv2_adam, info_mobilenetv2_adam] = trainNetwork(augimdsTrainTransfer, lgraph, optionsAdam);

classNames = categories(imdsTrain.Labels);

netCustom = net_mobilenetv2_adam;
save('mobilenetv2_adam_best_model.mat', 'netCustom', 'classNames', '-v7.3');
disp('En iyi MobileNetV2 Adam modeli başarıyla kaydedildi.');

hyperParams = table;
hyperParams.LossFunction = "categorical crossentropy";
hyperParams.NumEpochs = optionsAdam.MaxEpochs;
hyperParams.BatchSize = optionsAdam.MiniBatchSize;
hyperParams.OptimizationAlgorithm = "adam";
hyperParams.LearningRate = optionsAdam.InitialLearnRate;
hyperParams.ActivationFunction = "softmax";
hyperParams.NumDenseUnits = "MobileNetV2 son katmanına adapte edildi";
hyperParams.ValidationFraction = 0.2;

writetable(hyperParams, 'mobilenetv2_adam_hyperparameters.csv');
disp('En iyi MobileNetV2 Adam hiperparametre tablosu başarıyla kaydedildi.');

%}

%{

%% Adım 6.2: MobileNetV2 - Adam Eğitilmiş Model ile Test ve Performans Değerlendirmesi

if isfile('mobilenetv2_adam_best_model.mat')
    load('mobilenetv2_adam_best_model.mat', 'netCustom', 'classNames');
else
    error('mobilenetv2_adam_best_model.mat dosyası bulunamadı. Lütfen önce eğitim adımını tamamlayın.');
end

inputSize = netCustom.Layers(1).InputSize;
augimdsTestTransfer = augmentedImageDatastore(inputSize(1:2), imdsTest);

predictedLabels_mobilenetv2_adam = classify(netCustom, augimdsTestTransfer);
trueLabels = imdsTest.Labels;

confMat_mobilenetv2_adam = confusionmat(trueLabels, predictedLabels_mobilenetv2_adam);

TP = confMat_mobilenetv2_adam(2,2);
FP = confMat_mobilenetv2_adam(1,2);
FN = confMat_mobilenetv2_adam(2,1);
TN = confMat_mobilenetv2_adam(1,1);

accuracy_mobilenetv2_adam = (TP + TN) / (TP + TN + FP + FN) * 100;
precision_mobilenetv2_adam = (TP / (TP + FP)) * 100;
sensitivity_mobilenetv2_adam = (TP / (TP + FN)) * 100;
f1_score_mobilenetv2_adam = 2 * (precision_mobilenetv2_adam * sensitivity_mobilenetv2_adam) / (precision_mobilenetv2_adam + sensitivity_mobilenetv2_adam);

fprintf('MobileNetV2 Adam Test Performansı:\n');
fprintf('Accuracy: %.2f%%\n', accuracy_mobilenetv2_adam);
fprintf('Precision: %.2f%%\n', precision_mobilenetv2_adam);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity_mobilenetv2_adam);
fprintf('F1-Score: %.2f%%\n', f1_score_mobilenetv2_adam);

figure;
confusionchart(trueLabels, predictedLabels_mobilenetv2_adam);
title('Confusion Matrix - MobileNetV2 Adam (Test)');
grid on;

%}

%{
%% Adım 6.3: MobileNetV2 - RMSprop Optimizasyonu ile Eğitim ve Hiperparametre Kaydı

net = mobilenetv2;

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_output')
];

lgraph = removeLayers(lgraph, {'Logits', 'Logits_softmax', 'ClassificationLayer_Logits'});

lgraph = addLayers(lgraph, newLayers);

lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

optionsRMSprop = trainingOptions('rmsprop', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 5);

augimdsTrainTransfer = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augimdsValidationTransfer = augmentedImageDatastore(inputSize(1:2), imdsValidation);

[net_mobilenetv2_rmsprop, info_mobilenetv2_rmsprop] = trainNetwork(augimdsTrainTransfer, lgraph, optionsRMSprop);

classNames = categories(imdsTrain.Labels);

netCustom = net_mobilenetv2_rmsprop;
save('mobilenetv2_rmsprop_best_model.mat', 'netCustom', 'classNames', '-v7.3');
disp('En iyi MobileNetV2 RMSprop modeli başarıyla kaydedildi.');

hyperParams = table;
hyperParams.LossFunction = "categorical crossentropy";
hyperParams.NumEpochs = optionsRMSprop.MaxEpochs;
hyperParams.BatchSize = optionsRMSprop.MiniBatchSize;
hyperParams.OptimizationAlgorithm = "rmsprop";
hyperParams.LearningRate = optionsRMSprop.InitialLearnRate;
hyperParams.ActivationFunction = "softmax";
hyperParams.NumDenseUnits = "MobileNetV2 son katmanına adapte edildi";
hyperParams.ValidationFraction = 0.2;

writetable(hyperParams, 'mobilenetv2_rmsprop_hyperparameters.csv');
disp('En iyi MobileNetV2 RMSprop hiperparametre tablosu başarıyla kaydedildi.');

%}

%{

%% Adım 6.4: MobileNetV2 - RMSprop Eğitilmiş Model ile Test ve Performans Değerlendirmesi

if isfile('mobilenetv2_rmsprop_best_model.mat')
    load('mobilenetv2_rmsprop_best_model.mat', 'netCustom', 'classNames');
else
    error('mobilenetv2_rmsprop_best_model.mat dosyası bulunamadı. Lütfen önce eğitim adımını tamamlayın.');
end

inputSize = netCustom.Layers(1).InputSize; 
augimdsTestTransfer = augmentedImageDatastore(inputSize(1:2), imdsTest);

predictedLabels_mobilenetv2_rmsprop = classify(netCustom, augimdsTestTransfer);
trueLabels = imdsTest.Labels;

confMat_mobilenetv2_rmsprop = confusionmat(trueLabels, predictedLabels_mobilenetv2_rmsprop);

TP = confMat_mobilenetv2_rmsprop(2,2);
FP = confMat_mobilenetv2_rmsprop(1,2);
FN = confMat_mobilenetv2_rmsprop(2,1);
TN = confMat_mobilenetv2_rmsprop(1,1);

accuracy_mobilenetv2_rmsprop = (TP + TN) / (TP + TN + FP + FN) * 100;
precision_mobilenetv2_rmsprop = (TP / (TP + FP)) * 100;
sensitivity_mobilenetv2_rmsprop = (TP / (TP + FN)) * 100;
f1_score_mobilenetv2_rmsprop = 2 * (precision_mobilenetv2_rmsprop * sensitivity_mobilenetv2_rmsprop) / (precision_mobilenetv2_rmsprop + sensitivity_mobilenetv2_rmsprop);

fprintf('MobileNetV2 RMSprop Test Performansı:\n');
fprintf('Accuracy: %.2f%%\n', accuracy_mobilenetv2_rmsprop);
fprintf('Precision: %.2f%%\n', precision_mobilenetv2_rmsprop);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity_mobilenetv2_rmsprop);
fprintf('F1-Score: %.2f%%\n', f1_score_mobilenetv2_rmsprop);

figure;
confusionchart(trueLabels, predictedLabels_mobilenetv2_rmsprop);
title('Confusion Matrix - MobileNetV2 RMSprop (Test)');
grid on;

%}

%{
save('cnn_adam_trainNetworkProject.mat', 'net_cnn_adam', 'info_adam', 'optionsAdam', '-v7.3');
disp('CNN Adam trainNetworkProject dosyası başarıyla kaydedildi.');

save('cnn_rmsprop_trainNetworkProject.mat', 'net_cnn_rmsprop', 'info_rmsprop', 'optionsRMSprop', '-v7.3');
disp('CNN RMSprop trainNetworkProject dosyası başarıyla kaydedildi.');

save('cnn_sgdm_trainNetworkProject.mat', 'net_cnn_sgdm', 'info_sgdm', 'optionsSGDM', '-v7.3');
disp('CNN SGDM trainNetworkProject dosyası başarıyla kaydedildi.');

%}


%{
%% Adım 6.5: MobileNetV2 - SGDM Optimizasyonu ile Eğitim ve Hiperparametre Kaydı

net = mobilenetv2;

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels)); 

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_output')
];

lgraph = removeLayers(lgraph, {'Logits', 'Logits_softmax', 'ClassificationLayer_Logits'});

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

optionsSGDM = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-4, ...
    'Momentum', 0.9, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ValidationPatience', 5);

augimdsTrainTransfer = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augimdsValidationTransfer = augmentedImageDatastore(inputSize(1:2), imdsValidation);

[net_mobilenetv2_sgdm, info_mobilenetv2_sgdm] = trainNetwork(augimdsTrainTransfer, lgraph, optionsSGDM);

classNames = categories(imdsTrain.Labels);

netCustom = net_mobilenetv2_sgdm;
save('mobilenetv2_sgdm_best_model.mat', 'netCustom', 'classNames', '-v7.3');
disp('En iyi MobileNetV2 SGDM modeli başarıyla kaydedildi.');

save('mobilenetv2_sgdm_trainNetworkProject.mat', 'net_mobilenetv2_sgdm', 'info_mobilenetv2_sgdm', 'optionsSGDM', '-v7.3');
disp('MobileNetV2 SGDM eğitim süreci başarıyla kaydedildi.');

hyperParams = table;
hyperParams.LossFunction = "categorical crossentropy";
hyperParams.NumEpochs = optionsSGDM.MaxEpochs;
hyperParams.BatchSize = optionsSGDM.MiniBatchSize;
hyperParams.OptimizationAlgorithm = "sgdm";
hyperParams.LearningRate = optionsSGDM.InitialLearnRate;
hyperParams.ActivationFunction = "softmax";
hyperParams.NumDenseUnits = "MobileNetV2 son katmanına adapte edildi";
hyperParams.ValidationFraction = 0.2;

writetable(hyperParams, 'mobilenetv2_sgdm_hyperparameters.csv');
disp('En iyi MobileNetV2 SGDM hiperparametre tablosu başarıyla kaydedildi.');

%}

%{
%% Adım 6.6: MobileNetV2 SGDM - Eğitilmiş Model ile Test ve Test Sonuçlarını Kaydetme

if isfile('mobilenetv2_sgdm_best_model.mat')
    load('mobilenetv2_sgdm_best_model.mat', 'netCustom', 'classNames');
else
    error('mobilenetv2_sgdm_best_model.mat dosyası bulunamadı.');
end

inputSize = netCustom.Layers(1).InputSize;

augimdsTestResized = augmentedImageDatastore(inputSize(1:2), imdsTest);

predictedLabels_sgdm = classify(netCustom, augimdsTestResized);
trueLabels = imdsTest.Labels;

confMat_sgdm = confusionmat(trueLabels, predictedLabels_sgdm);

TP = confMat_sgdm(2,2);
FP = confMat_sgdm(1,2);
FN = confMat_sgdm(2,1);
TN = confMat_sgdm(1,1);

accuracy_sgdm = (TP + TN) / (TP + TN + FP + FN) * 100;
precision_sgdm = (TP / (TP + FP)) * 100;
sensitivity_sgdm = (TP / (TP + FN)) * 100;
f1_score_sgdm = 2 * (precision_sgdm * sensitivity_sgdm) / (precision_sgdm + sensitivity_sgdm);

fprintf('MobileNetV2 SGDM Test Sonuçları:\n');
fprintf('Accuracy: %.2f%%\n', accuracy_sgdm);
fprintf('Precision: %.2f%%\n', precision_sgdm);
fprintf('Sensitivity (Recall): %.2f%%\n', sensitivity_sgdm);
fprintf('F1-Score: %.2f%%\n', f1_score_sgdm);

figure;
confusionchart(trueLabels, predictedLabels_sgdm);
title('Confusion Matrix - MobileNetV2 SGDM');

save('mobilenetv2_sgdm_test_results.mat', ...
    'accuracy_sgdm', 'precision_sgdm', 'sensitivity_sgdm', 'f1_score_sgdm', 'confMat_sgdm');
disp('MobileNetV2 SGDM test sonuçları başarıyla kaydedildi.');

%}

%{
%% Tüm Eğitim Modellerini ve Eğitim Bilgilerini trainNetworkProject Klasörüne Kaydetme

saveFolder = fullfile(projectRoot, 'trainNetworkProject');

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

disp('Model kayıt işlemleri başladı...');

save(fullfile(saveFolder, 'cnn_adam_trainNetworkProject.mat'), ...
    'net_cnn_adam', 'info_adam', 'optionsAdam', '-v7.3');
disp('CNN Adam modeli kaydedildi.');

save(fullfile(saveFolder, 'cnn_rmsprop_trainNetworkProject.mat'), ...
    'net_cnn_rmsprop', 'info_rmsprop', 'optionsRMSprop', '-v7.3');
disp('CNN RMSprop modeli kaydedildi.');

save(fullfile(saveFolder, 'cnn_sgdm_trainNetworkProject.mat'), ...
    'net_cnn_sgdm', 'info_sgdm', 'optionsSGDM', '-v7.3');
disp('CNN SGDM modeli kaydedildi.');

save(fullfile(saveFolder, 'mobilenetv2_adam_trainNetworkProject.mat'), ...
    'net_mobilenetv2_adam', 'info_mobilenetv2_adam', 'optionsAdam', '-v7.3');
disp('MobileNetV2 Adam modeli kaydedildi.');

save(fullfile(saveFolder, 'mobilenetv2_rmsprop_trainNetworkProject.mat'), ...
    'net_mobilenetv2_rmsprop', 'info_mobilenetv2_rmsprop', 'optionsRMSprop', '-v7.3');
disp('MobileNetV2 RMSprop modeli kaydedildi.');

save(fullfile(saveFolder, 'mobilenetv2_sgdm_trainNetworkProject.mat'), ...
    'net_mobilenetv2_sgdm', 'info_mobilenetv2_sgdm', 'optionsSGDM', '-v7.3');
disp('MobileNetV2 SGDM modeli kaydedildi.');

disp('Tüm model kayıt işlemleri başarıyla tamamlandı.');

%}


%{
%% Adım 7: MobileNetV2 - Adam, RMSprop ve SGDM Test Accuracy Karşılaştırması

accuracy_mobilenetv2_adam = 0.9595;    
accuracy_mobilenetv2_rmsprop = 0.9324;
accuracy_mobilenetv2_sgdm = 0.9459;   

accuracies_mobilenetv2 = [accuracy_mobilenetv2_adam, accuracy_mobilenetv2_rmsprop, accuracy_mobilenetv2_sgdm] * 100;
optimizers = categorical({'Adam', 'RMSprop', 'SGDM'});

figure;
bar(optimizers, accuracies_mobilenetv2);
ylabel('Test Accuracy (%)');
xlabel('Optimizer');
ylim([0 100]);
title('MobileNetV2 - Adam, RMSprop, SGDM Test Accuracy Karşılaştırması');
grid on;

for i = 1:length(accuracies_mobilenetv2)
    text(i, accuracies_mobilenetv2(i)+1, sprintf('%.2f%%', accuracies_mobilenetv2(i)), 'HorizontalAlignment', 'center');
end

savefig(fullfile(saveFolder, 'mobilenetv2_optimizer_accuracy_comparison.fig'));
disp('MobileNetV2 optimizer test accuracy grafiği .fig olarak kaydedildi.');

%}

%{
%% Adım 7.1: MobileNetV2 - Adam, RMSprop ve SGDM Eğitim Accuracy Karşılaştırması

train_accuracy_mobilenetv2_adam = 0.9820;   
train_accuracy_mobilenetv2_rmsprop = 0.9615;
train_accuracy_mobilenetv2_sgdm = 0.9687;   

train_accuracies_mobilenetv2 = [train_accuracy_mobilenetv2_adam, train_accuracy_mobilenetv2_rmsprop, train_accuracy_mobilenetv2_sgdm] * 100;
optimizers = categorical({'Adam', 'RMSprop', 'SGDM'});

figure;
bar(optimizers, train_accuracies_mobilenetv2);
ylabel('Training Accuracy (%)');
xlabel('Optimizer');
ylim([0 100]);
title('MobileNetV2 - Adam, RMSprop, SGDM Training Accuracy Karşılaştırması');
grid on;

for i = 1:length(train_accuracies_mobilenetv2)
    text(i, train_accuracies_mobilenetv2(i)+1, sprintf('%.2f%%', train_accuracies_mobilenetv2(i)), 'HorizontalAlignment', 'center');
end

savefig(fullfile(saveFolder, 'mobilenetv2_optimizer_training_accuracy_comparison.fig'));
disp('MobileNetV2 optimizer training accuracy grafiği .fig olarak kaydedildi.');

%}

%{
%% Adım 8: CNN ve MobileNetV2 Test Accuracy Karşılaştırması

accuracy_cnn_adam = 0.9041;  
accuracy_cnn_rmsprop = 0.8815; 
accuracy_cnn_sgdm = 0.8697;   

accuracy_mobilenetv2_adam = 0.9595;    
accuracy_mobilenetv2_rmsprop = 0.9324; 
accuracy_mobilenetv2_sgdm = 0.9459;  

modelNames = categorical({
    'CNN Adam', 'CNN RMSprop', 'CNN SGDM', ...
    'MobileNetV2 Adam', 'MobileNetV2 RMSprop', 'MobileNetV2 SGDM'
    });

accuracies = [accuracy_cnn_adam, accuracy_cnn_rmsprop, accuracy_cnn_sgdm, ...
              accuracy_mobilenetv2_adam, accuracy_mobilenetv2_rmsprop, accuracy_mobilenetv2_sgdm] * 100;

figure;
bar(modelNames, accuracies);
ylabel('Test Accuracy (%)');
xlabel('Model - Optimizer');
ylim([0 100]);
title('CNN ve MobileNetV2 Test Accuracy Karşılaştırması');
grid on;

for i = 1:length(accuracies)
    text(i, accuracies(i)+1, sprintf('%.2f%%', accuracies(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

savefig(fullfile(saveFolder, 'cnn_vs_mobilenetv2_accuracy_comparison.fig'));
disp('CNN vs MobileNetV2 accuracy karşılaştırma grafiği .fig olarak kaydedildi.');

%}

%{

%% Adım 8.1: CNN ve MobileNetV2 Eğitim Accuracy Karşılaştırması

train_accuracy_cnn_adam = 0.9310;    
train_accuracy_cnn_rmsprop = 0.9025; 
train_accuracy_cnn_sgdm = 0.9158;   

train_accuracy_mobilenetv2_adam = 0.9820;    
train_accuracy_mobilenetv2_rmsprop = 0.9615;
train_accuracy_mobilenetv2_sgdm = 0.9687;   

modelNames = categorical({
    'CNN Adam', 'CNN RMSprop', 'CNN SGDM', ...
    'MobileNetV2 Adam', 'MobileNetV2 RMSprop', 'MobileNetV2 SGDM'
    });

train_accuracies = [train_accuracy_cnn_adam, train_accuracy_cnn_rmsprop, train_accuracy_cnn_sgdm, ...
                    train_accuracy_mobilenetv2_adam, train_accuracy_mobilenetv2_rmsprop, train_accuracy_mobilenetv2_sgdm] * 100;

figure;
bar(modelNames, train_accuracies);
ylabel('Training Accuracy (%)');
xlabel('Model - Optimizer');
ylim([0 100]);
title('CNN ve MobileNetV2 Eğitim Accuracy Karşılaştırması');
grid on;

for i = 1:length(train_accuracies)
    text(i, train_accuracies(i)+1, sprintf('%.2f%%', train_accuracies(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

savefig(fullfile(saveFolder, 'cnn_vs_mobilenetv2_training_accuracy_comparison.fig'));
disp('CNN vs MobileNetV2 eğitim accuracy karşılaştırma grafiği .fig olarak kaydedildi.');

%}

