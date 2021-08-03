# Converting a Paddle* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle}

A summary of the steps for optimizing and deploying a model that was trained with Paddle\*:

1. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for Paddle\*.
2. [Convert a Paddle\* Model](#Convert_From_Paddle) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values
3. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided Inference Engine [sample applications](../../../IE_DG/Samples_Overview.md) // sample?
4. [Integrate](../../../IE_DG/Samples_Overview.md) the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in your application to deploy the model in the target environment

## Supported Topologies

* **Classification models:**
	* AlexNet
	* VGG-16, VGG-19
	* SqueezeNet v1.0, SqueezeNet v1.1
	* ResNet-50, ResNet-101, Res-Net-152
	* Inception v1, Inception v2, Inception v3, Inception v4
	* CaffeNet
	* MobileNet
	* Squeeze-and-Excitation Networks: SE-BN-Inception, SE-Resnet-101, SE-ResNet-152, SE-ResNet-50, SE-ResNeXt-101, SE-ResNeXt-50
	* ShuffleNet v2

* **Object detection models:**
	* SSD300-VGG16, SSD500-VGG16
	* Faster-RCNN
	* RefineDet (MYRIAD plugin only)

* **Face detection models:**
	* VGG Face
    * SSH: Single Stage Headless Face Detector

* **Semantic segmentation models:**
	* FCN8

> **NOTE:** It is necessary to specify mean and scale values for most of the Caffe\* models to convert them with the Model Optimizer. The exact values should be determined separately for each model. For example, for Caffe\* models trained on ImageNet, the mean values usually are `123.68`, `116.779`, `103.939` for blue, green and red channels respectively. The scale value is usually `127.5`. Refer to [Framework-agnostic parameters](Converting_Model_General.md) for the information on how to specify mean and scale values.

## Convert a Paddle* Model <a name="Convert_From_Paddle"></a>

To convert a Paddle\* model:

1. Go to the `$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer` directory.
2. Use the `mo.py` script to simply convert a model, specifying the path to the input model `.pdmodel` file and the path to an output directory with write permissions:
```sh
python3 mo.py --input_model <INPUT_MODEL>.pdmodel --output_dir <OUTPUT_MODEL_DIR>
```

Two groups of parameters are available to convert your model:

* [Framework-agnostic parameters](Converting_Model_General.md): These parameters are used to convert a model trained with any supported framework.
> **NOTE:** `--scale`, `--scale_values`, `--mean_values`, `--mean_file` are unavailable in mo_paddle
* [Paddle-specific parameters](#paddle_specific_conversion_params): Parameters used to convert only Paddle\* models.


## Custom Layer Definition ??

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

## Supported Paddle\* Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to issues like typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ has instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary

In this document, you learned:

* Basic information about how the Model Optimizer works with Paddle\* models
* Which Paddle\* models are supported
* How to convert a trained Paddle\* model using the Model Optimizer with both framework-agnostic and Paddle-specific command-line options
