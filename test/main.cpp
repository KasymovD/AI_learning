#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <iomanip>

using namespace cv;
using namespace cv::dnn;
using namespace std;
namespace fs = std::filesystem;

Net buildCNNModel() {
    Net model;
    
    model.addLayerToPrev("input", "Input", LayerParams());

    LayerParams convParams;
    convParams.set("kernel_size", 3);
    convParams.set("num_output", 32);
    convParams.set("pad", 1);
    model.addLayerToPrev("conv1", "Convolution", convParams);

    model.addLayerToPrev("relu1", "ReLU", LayerParams());

    LayerParams poolParams;
    poolParams.set("kernel_size", 2);
    poolParams.set("stride", 2);
    model.addLayerToPrev("pool1", "Pooling", poolParams);

    LayerParams fcParams;
    fcParams.set("num_output", 10);
    model.addLayerToPrev("fc", "InnerProduct", fcParams);

    model.addLayerToPrev("softmax", "Softmax", LayerParams());

    return model;
}

Mat preprocessImage(const string& path) {
    Mat img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw runtime_error("Failed to load image: " + path);
    }

    resize(img, img, Size(28, 28));
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    return blobFromImage(img);
}

void displayResults(const string& imagePath, int predictedClass, double confidence) {
    cout << left << setw(20) << "Image Path:" << imagePath << endl;
    cout << left << setw(20) << "Predicted Class:" << predictedClass << endl;
    cout << left << setw(20) << "Confidence:" << fixed << setprecision(4) << confidence << endl;
    cout << string(40, '-') << endl;
}

void classify(Net& model, const vector<string>& images) {
    for (const auto& path : images) {
        try {
            Mat inputBlob = preprocessImage(path);
            model.setInput(inputBlob);
            Mat output = model.forward();

            Point classIdPoint;
            double confidence;
            minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
            displayResults(path, classIdPoint.x, confidence);
        } catch (const exception& ex) {
            cerr << "Error: " << ex.what() << endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path1> [<image_path2> ...]" << endl;
        return -1;
    }

    vector<string> imagePaths(argv + 1, argv + argc);
    Net model = buildCNNModel();
    classify(model, imagePaths);

    return 0;
}
