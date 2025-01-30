#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace cv::dnn;
using namespace std;
namespace fs = std::filesystem;

Net createCNN() {
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

    return cnn;
}

Mat preprocessImage(const string& imagePath) {
    Mat inputImage = imread(imagePath, IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        throw runtime_error("Error loading image: " + imagePath);
    }

    resize(inputImage, inputImage, Size(28, 28));
    inputImage.convertTo(inputImage, CV_32F, 1.0 / 255.0);
    return blobFromImage(inputImage);
}

void classifyImages(Net& cnn, const vector<string>& imagePaths) {
    for (const auto& imagePath : imagePaths) {
        try {
            Mat blob = preprocessImage(imagePath);
            cnn.setInput(blob);
            Mat output = cnn.forward();

            Point classIdPoint;
            double confidence;
            minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
            int predictedClass = classIdPoint.x;

            cout << "Image: " << imagePath << endl;
            cout << "Predicted class: " << predictedClass << endl;
            cout << "Confidence: " << confidence << endl << endl;
        } catch (const exception& e) {
            cerr << e.what() << endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path1> [<image_path2> ...]" << endl;
        return -1;
    }

    vector<string> imagePaths;
    for (int i = 1; i < argc; ++i) {
        imagePaths.push_back(argv[i]);
    }

    Net cnn = createCNN();
    classifyImages(cnn, imagePaths);

    return 0;
}
