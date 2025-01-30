#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    Mat inputImage = imread("image.png", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cerr << "Image loading error!" << endl;
        return -1;
    }

    resize(inputImage, inputImage, Size(28, 28));
    inputImage.convertTo(inputImage, CV_32F, 1.0 / 255.0);

    Net cnn;

    cnn.addLayerToPrev("input", "Input", LayerParams());

    LayerParams conv1Params;
    conv1Params.set("kernel_size", 3);
    conv1Params.set("num_output", 16);
    conv1Params.set("pad", 1);
    cnn.addLayerToPrev("conv1", "Convolution", conv1Params);

    cnn.addLayerToPrev("relu1", "ReLU", LayerParams());

    LayerParams pool1Params;
    pool1Params.set("kernel_size", 2);
    pool1Params.set("stride", 2);
    cnn.addLayerToPrev("pool1", "Pooling", pool1Params);

    LayerParams fcParams;
    fcParams.set("num_output", 10);
    cnn.addLayerToPrev("fc", "InnerProduct", fcParams);

    cnn.addLayerToPrev("softmax", "Softmax", LayerParams());

    Mat blob = blobFromImage(inputImage);

    cnn.setInput(blob);
    Mat output = cnn.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
    int predictedClass = classIdPoint.x;

    cout << "Predicted class: " << predictedClass << endl;
    cout << "Confidence: " << confidence << endl;

    return 0;
}
