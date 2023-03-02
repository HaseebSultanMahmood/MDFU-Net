%%%%%%% MultiScale Dilated Features Upsampling Net (MDFU-Net) (No- augmentation)
% this is the 2nd proposed net with Weighted Cross entropy loss function

%%%%%%%% 
function [lgraph, ModelName] = fn_MDFU_Net_WCE(numClasses,classNames,inverseFrequency,params)
disp('multiscale dilated features upsampling (MDFU_Net_WCE) is selected for this experiment');
ModelName = 'MDFU_Net_WCE';
lgraph = layerGraph();

tempLayers = imageInputLayer([304 304 3],"Name","input_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001,"Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0,"Bias",params.res2a_branch1.Bias,"Weights",params.res2a_branch1.Weights)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001,"Offset",params.bn2a_branch1.Offset,"Scale",params.bn2a_branch1.Scale,"TrainedMean",params.bn2a_branch1.TrainedMean,"TrainedVariance",params.bn2a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001,"Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001,"Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0,"Bias",params.res2a_branch2c.Bias,"Weights",params.res2a_branch2c.Weights)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001,"Offset",params.bn2a_branch2c.Offset,"Scale",params.bn2a_branch2c.Scale,"TrainedMean",params.bn2a_branch2c.TrainedMean,"TrainedVariance",params.bn2a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001,"Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001,"Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0,"Bias",params.res2b_branch2c.Bias,"Weights",params.res2b_branch2c.Weights)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001,"Offset",params.bn2b_branch2c.Offset,"Scale",params.bn2b_branch2c.Scale,"TrainedMean",params.bn2b_branch2c.TrainedMean,"TrainedVariance",params.bn2b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0,"Bias",params.res2c_branch2a.Bias,"Weights",params.res2c_branch2a.Weights)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001,"Offset",params.bn2c_branch2a.Offset,"Scale",params.bn2c_branch2a.Scale,"TrainedMean",params.bn2c_branch2a.TrainedMean,"TrainedVariance",params.bn2c_branch2a.TrainedVariance)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res2c_branch2b.Bias,"Weights",params.res2c_branch2b.Weights)
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001,"Offset",params.bn2c_branch2b.Offset,"Scale",params.bn2c_branch2b.Scale,"TrainedMean",params.bn2c_branch2b.TrainedMean,"TrainedVariance",params.bn2c_branch2b.TrainedVariance)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0,"Bias",params.res2c_branch2c.Bias,"Weights",params.res2c_branch2c.Weights)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001,"Offset",params.bn2c_branch2c.Offset,"Scale",params.bn2c_branch2c.Scale,"TrainedMean",params.bn2c_branch2c.TrainedMean,"TrainedVariance",params.bn2c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res3a_branch1.Bias,"Weights",params.res3a_branch1.Weights)
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001,"Offset",params.bn3a_branch1.Offset,"Scale",params.bn3a_branch1.Scale,"TrainedMean",params.bn3a_branch1.TrainedMean,"TrainedVariance",params.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res3a_branch2a.Bias,"Weights",params.res3a_branch2a.Weights)
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001,"Offset",params.bn3a_branch2a.Offset,"Scale",params.bn3a_branch2a.Scale,"TrainedMean",params.bn3a_branch2a.TrainedMean,"TrainedVariance",params.bn3a_branch2a.TrainedVariance)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001,"Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0,"Bias",params.res3a_branch2c.Bias,"Weights",params.res3a_branch2c.Weights)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001,"Offset",params.bn3a_branch2c.Offset,"Scale",params.bn3a_branch2c.Scale,"TrainedMean",params.bn3a_branch2c.TrainedMean,"TrainedVariance",params.bn3a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Bias",params.res3b_branch2a.Bias,"Weights",params.res3b_branch2a.Weights)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001,"Offset",params.bn3b_branch2a.Offset,"Scale",params.bn3b_branch2a.Scale,"TrainedMean",params.bn3b_branch2a.TrainedMean,"TrainedVariance",params.bn3b_branch2a.TrainedVariance)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3b_branch2b.Bias,"Weights",params.res3b_branch2b.Weights)
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001,"Offset",params.bn3b_branch2b.Offset,"Scale",params.bn3b_branch2b.Scale,"TrainedMean",params.bn3b_branch2b.TrainedMean,"TrainedVariance",params.bn3b_branch2b.TrainedVariance)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0,"Bias",params.res3b_branch2c.Bias,"Weights",params.res3b_branch2c.Weights)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001,"Offset",params.bn3b_branch2c.Offset,"Scale",params.bn3b_branch2c.Scale,"TrainedMean",params.bn3b_branch2c.TrainedMean,"TrainedVariance",params.bn3b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0,"Bias",params.res3c_branch2a.Bias,"Weights",params.res3c_branch2a.Weights)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001,"Offset",params.bn3c_branch2a.Offset,"Scale",params.bn3c_branch2a.Scale,"TrainedMean",params.bn3c_branch2a.TrainedMean,"TrainedVariance",params.bn3c_branch2a.TrainedVariance)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3c_branch2b.Bias,"Weights",params.res3c_branch2b.Weights)
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001,"Offset",params.bn3c_branch2b.Offset,"Scale",params.bn3c_branch2b.Scale,"TrainedMean",params.bn3c_branch2b.TrainedMean,"TrainedVariance",params.bn3c_branch2b.TrainedVariance)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0,"Bias",params.res3c_branch2c.Bias,"Weights",params.res3c_branch2c.Weights)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001,"Offset",params.bn3c_branch2c.Offset,"Scale",params.bn3c_branch2c.Scale,"TrainedMean",params.bn3c_branch2c.TrainedMean,"TrainedVariance",params.bn3c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0,"Bias",params.res3d_branch2a.Bias,"Weights",params.res3d_branch2a.Weights)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001,"Offset",params.bn3d_branch2a.Offset,"Scale",params.bn3d_branch2a.Scale,"TrainedMean",params.bn3d_branch2a.TrainedMean,"TrainedVariance",params.bn3d_branch2a.TrainedVariance)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3d_branch2b.Bias,"Weights",params.res3d_branch2b.Weights)
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001,"Offset",params.bn3d_branch2b.Offset,"Scale",params.bn3d_branch2b.Scale,"TrainedMean",params.bn3d_branch2b.TrainedMean,"TrainedVariance",params.bn3d_branch2b.TrainedVariance)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0,"Bias",params.res3d_branch2c.Bias,"Weights",params.res3d_branch2c.Weights)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001,"Offset",params.bn3d_branch2c.Offset,"Scale",params.bn3d_branch2c.Scale,"TrainedMean",params.bn3d_branch2c.TrainedMean,"TrainedVariance",params.bn3d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.res4a_branch1.Bias,"Weights",params.res4a_branch1.Weights)
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001,"Offset",params.bn4a_branch1.Offset,"Scale",params.bn4a_branch1.Scale,"TrainedMean",params.bn4a_branch1.TrainedMean,"TrainedVariance",params.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.res4a_branch2a.Bias,"Weights",params.res4a_branch2a.Weights)
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001,"Offset",params.bn4a_branch2a.Offset,"Scale",params.bn4a_branch2a.Scale,"TrainedMean",params.bn4a_branch2a.TrainedMean,"TrainedVariance",params.bn4a_branch2a.TrainedVariance)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001,"Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0,"Bias",params.res4a_branch2c.Bias,"Weights",params.res4a_branch2c.Weights)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001,"Offset",params.bn4a_branch2c.Offset,"Scale",params.bn4a_branch2c.Scale,"TrainedMean",params.bn4a_branch2c.TrainedMean,"TrainedVariance",params.bn4a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Bias",params.res4b_branch2a.Bias,"Weights",params.res4b_branch2a.Weights)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001,"Offset",params.bn4b_branch2a.Offset,"Scale",params.bn4b_branch2a.Scale,"TrainedMean",params.bn4b_branch2a.TrainedMean,"TrainedVariance",params.bn4b_branch2a.TrainedVariance)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4b_branch2b.Bias,"Weights",params.res4b_branch2b.Weights)
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001,"Offset",params.bn4b_branch2b.Offset,"Scale",params.bn4b_branch2b.Scale,"TrainedMean",params.bn4b_branch2b.TrainedMean,"TrainedVariance",params.bn4b_branch2b.TrainedVariance)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0,"Bias",params.res4b_branch2c.Bias,"Weights",params.res4b_branch2c.Weights)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001,"Offset",params.bn4b_branch2c.Offset,"Scale",params.bn4b_branch2c.Scale,"TrainedMean",params.bn4b_branch2c.TrainedMean,"TrainedVariance",params.bn4b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0,"Bias",params.res4c_branch2a.Bias,"Weights",params.res4c_branch2a.Weights)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001,"Offset",params.bn4c_branch2a.Offset,"Scale",params.bn4c_branch2a.Scale,"TrainedMean",params.bn4c_branch2a.TrainedMean,"TrainedVariance",params.bn4c_branch2a.TrainedVariance)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4c_branch2b.Bias,"Weights",params.res4c_branch2b.Weights)
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001,"Offset",params.bn4c_branch2b.Offset,"Scale",params.bn4c_branch2b.Scale,"TrainedMean",params.bn4c_branch2b.TrainedMean,"TrainedVariance",params.bn4c_branch2b.TrainedVariance)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0,"Bias",params.res4c_branch2c.Bias,"Weights",params.res4c_branch2c.Weights)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001,"Offset",params.bn4c_branch2c.Offset,"Scale",params.bn4c_branch2c.Scale,"TrainedMean",params.bn4c_branch2c.TrainedMean,"TrainedVariance",params.bn4c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0,"Bias",params.res4d_branch2a.Bias,"Weights",params.res4d_branch2a.Weights)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001,"Offset",params.bn4d_branch2a.Offset,"Scale",params.bn4d_branch2a.Scale,"TrainedMean",params.bn4d_branch2a.TrainedMean,"TrainedVariance",params.bn4d_branch2a.TrainedVariance)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4d_branch2b.Bias,"Weights",params.res4d_branch2b.Weights)
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001,"Offset",params.bn4d_branch2b.Offset,"Scale",params.bn4d_branch2b.Scale,"TrainedMean",params.bn4d_branch2b.TrainedMean,"TrainedVariance",params.bn4d_branch2b.TrainedVariance)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0,"Bias",params.res4d_branch2c.Bias,"Weights",params.res4d_branch2c.Weights)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001,"Offset",params.bn4d_branch2c.Offset,"Scale",params.bn4d_branch2c.Scale,"TrainedMean",params.bn4d_branch2c.TrainedMean,"TrainedVariance",params.bn4d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0,"Bias",params.res4e_branch2a.Bias,"Weights",params.res4e_branch2a.Weights)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001,"Offset",params.bn4e_branch2a.Offset,"Scale",params.bn4e_branch2a.Scale,"TrainedMean",params.bn4e_branch2a.TrainedMean,"TrainedVariance",params.bn4e_branch2a.TrainedVariance)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4e_branch2b.Bias,"Weights",params.res4e_branch2b.Weights)
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001,"Offset",params.bn4e_branch2b.Offset,"Scale",params.bn4e_branch2b.Scale,"TrainedMean",params.bn4e_branch2b.TrainedMean,"TrainedVariance",params.bn4e_branch2b.TrainedVariance)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0,"Bias",params.res4e_branch2c.Bias,"Weights",params.res4e_branch2c.Weights)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001,"Offset",params.bn4e_branch2c.Offset,"Scale",params.bn4e_branch2c.Scale,"TrainedMean",params.bn4e_branch2c.TrainedMean,"TrainedVariance",params.bn4e_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0,"Bias",params.res4f_branch2a.Bias,"Weights",params.res4f_branch2a.Weights)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001,"Offset",params.bn4f_branch2a.Offset,"Scale",params.bn4f_branch2a.Scale,"TrainedMean",params.bn4f_branch2a.TrainedMean,"TrainedVariance",params.bn4f_branch2a.TrainedVariance)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4f_branch2b.Bias,"Weights",params.res4f_branch2b.Weights)
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001,"Offset",params.bn4f_branch2b.Offset,"Scale",params.bn4f_branch2b.Scale,"TrainedMean",params.bn4f_branch2b.TrainedMean,"TrainedVariance",params.bn4f_branch2b.TrainedVariance)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0,"Bias",params.res4f_branch2c.Bias,"Weights",params.res4f_branch2c.Weights)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001,"Offset",params.bn4f_branch2c.Offset,"Scale",params.bn4f_branch2c.Scale,"TrainedMean",params.bn4f_branch2c.TrainedMean,"TrainedVariance",params.bn4f_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0,"Bias",params.res5a_branch1.Bias,"Weights",params.res5a_branch1.Weights)
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001,"Offset",params.bn5a_branch1.Offset,"Scale",params.bn5a_branch1.Scale,"TrainedMean",params.bn5a_branch1.TrainedMean,"TrainedVariance",params.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Bias",params.res5a_branch2a.Bias,"Weights",params.res5a_branch2a.Weights)
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001,"Offset",params.bn5a_branch2a.Offset,"Scale",params.bn5a_branch2a.Scale,"TrainedMean",params.bn5a_branch2a.TrainedMean,"TrainedVariance",params.bn5a_branch2a.TrainedVariance)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same","Bias",params.res5a_branch2b.Bias,"Weights",params.res5a_branch2b.Weights)
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001,"Offset",params.bn5a_branch2b.Offset,"Scale",params.bn5a_branch2b.Scale,"TrainedMean",params.bn5a_branch2b.TrainedMean,"TrainedVariance",params.bn5a_branch2b.TrainedVariance)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0,"Bias",params.res5a_branch2c.Bias,"Weights",params.res5a_branch2c.Weights)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001,"Offset",params.bn5a_branch2c.Offset,"Scale",params.bn5a_branch2c.Scale,"TrainedMean",params.bn5a_branch2c.TrainedMean,"TrainedVariance",params.bn5a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Bias",params.res5b_branch2a.Bias,"Weights",params.res5b_branch2a.Weights)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001,"Offset",params.bn5b_branch2a.Offset,"Scale",params.bn5b_branch2a.Scale,"TrainedMean",params.bn5b_branch2a.TrainedMean,"TrainedVariance",params.bn5b_branch2a.TrainedVariance)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same","Bias",params.res5b_branch2b.Bias,"Weights",params.res5b_branch2b.Weights)
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001,"Offset",params.bn5b_branch2b.Offset,"Scale",params.bn5b_branch2b.Scale,"TrainedMean",params.bn5b_branch2b.TrainedMean,"TrainedVariance",params.bn5b_branch2b.TrainedVariance)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0,"Bias",params.res5b_branch2c.Bias,"Weights",params.res5b_branch2c.Weights)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001,"Offset",params.bn5b_branch2c.Offset,"Scale",params.bn5b_branch2c.Scale,"TrainedMean",params.bn5b_branch2c.TrainedMean,"TrainedVariance",params.bn5b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

% tempLayers = [
%     convolution2dLayer([1 1],48,"Name","dec_c2","BiasLearnRateFactor",0,"WeightLearnRateFactor",10,"Bias",params.dec_c2.Bias,"Weights",params.dec_c2.Weights)
%     batchNormalizationLayer("Name","dec_bn2","Offset",params.dec_bn2.Offset,"Scale",params.dec_bn2.Scale,"TrainedMean",params.dec_bn2.TrainedMean,"TrainedVariance",params.dec_bn2.TrainedVariance)
%     reluLayer("Name","dec_relu2")];
% lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0,"Bias",params.res5c_branch2a.Bias,"Weights",params.res5c_branch2a.Weights)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001,"Offset",params.bn5c_branch2a.Offset,"Scale",params.bn5c_branch2a.Scale,"TrainedMean",params.bn5c_branch2a.TrainedMean,"TrainedVariance",params.bn5c_branch2a.TrainedVariance)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same","Bias",params.res5c_branch2b.Bias,"Weights",params.res5c_branch2b.Weights)
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001,"Offset",params.bn5c_branch2b.Offset,"Scale",params.bn5c_branch2b.Scale,"TrainedMean",params.bn5c_branch2b.TrainedMean,"TrainedVariance",params.bn5c_branch2b.TrainedVariance)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0,"Bias",params.res5c_branch2c.Bias,"Weights",params.res5c_branch2c.Weights)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001,"Offset",params.bn5c_branch2c.Offset,"Scale",params.bn5c_branch2c.Scale,"TrainedMean",params.bn5c_branch2c.TrainedMean,"TrainedVariance",params.bn5c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")];
lgraph = addLayers(lgraph,tempLayers);

%%%%%%%% % new Artrous block 1 (encoder side) after res 2c (before res 3a)
 tempLayers = [ 
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_1","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_1")
    reluLayer("Name","new_aspp_relu_1")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_2")
    batchNormalizationLayer("Name","new_aspp_bn_2")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_3","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_3")
    reluLayer("Name","new_aspp_relu_2")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_4")
    batchNormalizationLayer("Name","new_aspp_bn_4")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_5","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_5")
    reluLayer("Name","new_aspp_relu_3")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_6")
    batchNormalizationLayer("Name","new_aspp_bn_6")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_7","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_7")
    reluLayer("Name","new_aspp_relu_4")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_8")
    batchNormalizationLayer("Name","new_aspp_bn_8")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = depthConcatenationLayer(4,"Name","new_aspp_cat_1");
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([1 1],48,"Name","new_conv_9")
    batchNormalizationLayer("Name","new_bn_9")
    reluLayer("Name","new_relu_5")];
 lgraph = addLayers(lgraph,tempLayers);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%% % new Artrous block 2 (encoder side) after res 3d (before res 4a)
  tempLayers = [ 
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_10","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_10")
    reluLayer("Name","new_aspp_relu_6")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_11")
    batchNormalizationLayer("Name","new_aspp_bn_11")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_12","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_12")
    reluLayer("Name","new_aspp_relu_7")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_13")
    batchNormalizationLayer("Name","new_aspp_bn_13")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_14","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_14")
    reluLayer("Name","new_aspp_relu_8")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_15")
    batchNormalizationLayer("Name","new_aspp_bn_15")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_16","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_16")
    reluLayer("Name","new_aspp_relu_9")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_17")
    batchNormalizationLayer("Name","new_aspp_bn_17")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = depthConcatenationLayer(4,"Name","new_aspp_cat_2");
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([1 1],48,"Name","new_conv_18")
    batchNormalizationLayer("Name","new_bn_18")
    reluLayer("Name","new_relu_10")];
 lgraph = addLayers(lgraph,tempLayers); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%% % new Artrous block 3 (encoder side) after res 5c (before decoder)
tempLayers = [ 
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_19","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_19")
    reluLayer("Name","new_aspp_relu_11")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_20")
    batchNormalizationLayer("Name","new_aspp_bn_20")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_21","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_21")
    reluLayer("Name","new_aspp_relu_12")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_22")
    batchNormalizationLayer("Name","new_aspp_bn_22")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_23","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_23")
    reluLayer("Name","new_aspp_relu_13")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_24")
    batchNormalizationLayer("Name","new_aspp_bn_24")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = [
    convolution2dLayer([3 3],256,"Name","new_aspp_conv_25","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","new_aspp_bn_25")
    reluLayer("Name","new_aspp_relu_14")
    convolution2dLayer([1 1],256,"Name","new_aspp_conv_26")
    batchNormalizationLayer("Name","new_aspp_bn_26")];
   lgraph = addLayers(lgraph,tempLayers);

   tempLayers = depthConcatenationLayer(4,"Name","new_aspp_cat_3");
   lgraph = addLayers(lgraph,tempLayers);
 
 tempLayers = [
   convolution2dLayer([1 1],48,"Name","new_conv_27")
    batchNormalizationLayer("Name","new_bn_27")
    reluLayer("Name","new_relu_15")];
 lgraph = addLayers(lgraph,tempLayers); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% new upsampling blocks on decoder side

% after res 2c (before res 3a)
tempLayers = crop2dLayer("centercrop","Name","new_crop_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","new_cat_4")
    transposedConv2dLayer([4 4],256,"Name","new_tconv_1","Stride",[1 1],'Cropping','same')
    batchNormalizationLayer("Name","new_bn_28")
    reluLayer("Name","new_relu_16")
    transposedConv2dLayer([3 3],256,"Name","new_tconv_2","Stride",[1 1],'Cropping','same')
    batchNormalizationLayer("Name","new_bn_29")
    reluLayer("Name","new_relu_17")
    convolution2dLayer([1 1],numClasses,"Name","new_scorer_1")
    transposedConv2dLayer([8 8],numClasses,"Name","new_tconv_3","Stride",[4 4],"Cropping",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

% after res 3d (before res 4a)
tempLayers = crop2dLayer("centercrop","Name","new_crop_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","new_cat_5")
    transposedConv2dLayer([4 4],256,"Name","new_tconv_4","Stride",[2 2],'Cropping','same')
    batchNormalizationLayer("Name","new_bn_30")
    reluLayer("Name","new_relu_18")
    transposedConv2dLayer([3 3],256,"Name","new_tconv_5","Stride",[1 1],'Cropping','same')
    batchNormalizationLayer("Name","new_bn_31")
    reluLayer("Name","new_relu_19")
    convolution2dLayer([1 1],numClasses,"Name","new_scorer_2")
    transposedConv2dLayer([8 8],numClasses,"Name","new_tconv_6","Stride",[4 4],"Cropping",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

% after res 5c 
   tempLayers = transposedConv2dLayer([8 8],256,"Name","new_tconv_7","Cropping",[2 2 2 2],"Stride",[4 4]);
lgraph = addLayers(lgraph,tempLayers);

   tempLayers = crop2dLayer("centercrop","Name","new_crop_3");
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    depthConcatenationLayer(2,"Name","new_cat_6")
    transposedConv2dLayer([4 4],256,"Name","new_tconv_8","Stride",[4 4],'Cropping','same')
    batchNormalizationLayer("Name","new_bn_32")
    reluLayer("Name","new_relu_20")
    transposedConv2dLayer([3 3],256,"Name","new_tconv_9","Stride",[1 1],'Cropping','same')
    batchNormalizationLayer("Name","new_bn_33")
    reluLayer("Name","new_relu_21")
    convolution2dLayer([1 1],numClasses,"Name","new_scorer_3")
    transposedConv2dLayer([8 8],numClasses,"Name","new_tconv_10","Stride",[4 4],"Cropping",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tempLayers = depthConcatenationLayer(3,"Name","new_cat_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [convolution2dLayer([1 1],numClasses,"Name","new_scorer_4");
batchNormalizationLayer("Name","new_bn_34")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    crop2dLayer("centercrop","Name","old_crop_4")
    softmaxLayer("Name","softmax-out")
    pixelClassificationLayer("Name","classification","Classes",classNames,"ClassWeights",inverseFrequency)];
lgraph = addLayers(lgraph,tempLayers);


% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"input_1","conv1");
% lgraph = connectLayers(lgraph,"input_1","dec_crop2/ref");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
% lgraph = connectLayers(lgraph,"activation_10_relu","dec_c2");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
% lgraph = connectLayers(lgraph,"dec_relu2","dec_crop1/ref");
% lgraph = connectLayers(lgraph,"dec_relu2","dec_cat1/in1");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");
% lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_1");
% lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_2");
% lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_4");
% lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_3");
% lgraph = connectLayers(lgraph,"aspp_Relu_2","catAspp/in2");
% lgraph = connectLayers(lgraph,"aspp_Relu_4","catAspp/in4");
% lgraph = connectLayers(lgraph,"aspp_Relu_3","catAspp/in3");
% lgraph = connectLayers(lgraph,"aspp_Relu_1","catAspp/in1");
% lgraph = connectLayers(lgraph,"dec_upsample1","dec_crop1/in");
% lgraph = connectLayers(lgraph,"dec_crop1","dec_cat1/in2");
% lgraph = connectLayers(lgraph,"dec_upsample2","dec_crop2/in");

% connections after res 2c (before res 3a)
lgraph = connectLayers(lgraph,"activation_10_relu","new_aspp_conv_1");
lgraph = connectLayers(lgraph,"activation_10_relu","new_aspp_conv_3");
lgraph = connectLayers(lgraph,"activation_10_relu","new_aspp_conv_5");
lgraph = connectLayers(lgraph,"activation_10_relu","new_aspp_conv_7");
lgraph = connectLayers(lgraph,"new_aspp_bn_2","new_aspp_cat_1/in1");
lgraph = connectLayers(lgraph,"new_aspp_bn_4","new_aspp_cat_1/in2");
lgraph = connectLayers(lgraph,"new_aspp_bn_6","new_aspp_cat_1/in3");
lgraph = connectLayers(lgraph,"new_aspp_bn_8","new_aspp_cat_1/in4");
lgraph = connectLayers(lgraph,"new_aspp_cat_1","new_conv_9");

lgraph = connectLayers(lgraph,"new_tconv_7","new_crop_1/in");
lgraph = connectLayers(lgraph,"new_relu_5","new_crop_1/ref");
lgraph = connectLayers(lgraph,"new_relu_5","new_cat_4/in1");
lgraph = connectLayers(lgraph,"new_crop_1","new_cat_4/in2");
lgraph = connectLayers(lgraph,"new_tconv_3","new_cat_7/in1");

% connections after res 3d (before res 4a)
lgraph = connectLayers(lgraph,"activation_22_relu","new_aspp_conv_10");
lgraph = connectLayers(lgraph,"activation_22_relu","new_aspp_conv_12");
lgraph = connectLayers(lgraph,"activation_22_relu","new_aspp_conv_14");
lgraph = connectLayers(lgraph,"activation_22_relu","new_aspp_conv_16");
lgraph = connectLayers(lgraph,"new_aspp_bn_11","new_aspp_cat_2/in1");
lgraph = connectLayers(lgraph,"new_aspp_bn_13","new_aspp_cat_2/in2");
lgraph = connectLayers(lgraph,"new_aspp_bn_15","new_aspp_cat_2/in3");
lgraph = connectLayers(lgraph,"new_aspp_bn_17","new_aspp_cat_2/in4");
lgraph = connectLayers(lgraph,"new_aspp_cat_2","new_conv_18");

lgraph = connectLayers(lgraph,"new_tconv_7","new_crop_2/in");
lgraph = connectLayers(lgraph,"new_relu_10","new_crop_2/ref");
lgraph = connectLayers(lgraph,"new_relu_10","new_cat_5/in1");
lgraph = connectLayers(lgraph,"new_crop_2","new_cat_5/in2");
lgraph = connectLayers(lgraph,"new_tconv_6","new_cat_7/in2");

% connections after res 5c
lgraph = connectLayers(lgraph,"activation_49_relu","new_aspp_conv_19");
lgraph = connectLayers(lgraph,"activation_49_relu","new_aspp_conv_21");
lgraph = connectLayers(lgraph,"activation_49_relu","new_aspp_conv_23");
lgraph = connectLayers(lgraph,"activation_49_relu","new_aspp_conv_25");
lgraph = connectLayers(lgraph,"new_aspp_bn_20","new_aspp_cat_3/in1");
lgraph = connectLayers(lgraph,"new_aspp_bn_22","new_aspp_cat_3/in2");
lgraph = connectLayers(lgraph,"new_aspp_bn_24","new_aspp_cat_3/in3");
lgraph = connectLayers(lgraph,"new_aspp_bn_26","new_aspp_cat_3/in4");
lgraph = connectLayers(lgraph,"new_aspp_cat_3","new_conv_27");
lgraph = connectLayers(lgraph,"new_relu_15","new_tconv_7");

lgraph = connectLayers(lgraph,"new_tconv_7","new_crop_3/in");
lgraph = connectLayers(lgraph,"new_relu_15","new_crop_3/ref");
lgraph = connectLayers(lgraph,"new_relu_15","new_cat_6/in1");
lgraph = connectLayers(lgraph,"new_crop_3","new_cat_6/in2");
lgraph = connectLayers(lgraph,"new_tconv_10","new_cat_7/in3");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lgraph = connectLayers(lgraph,"new_cat_7","new_scorer_4");
lgraph = connectLayers(lgraph,"new_bn_34","old_crop_4/in");
lgraph = connectLayers(lgraph,"input_1","old_crop_4/ref");

end
