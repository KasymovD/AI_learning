const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

async function buildCNNModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

async function preprocessImage(imagePath) {
    const imageBuffer = fs.readFileSync(imagePath);
    const processedImage = await sharp(imageBuffer)
        .resize(28, 28)
        .greyscale()
        .raw()
        .toBuffer();

    const imageTensor = tf.tensor(processedImage, [28, 28, 1], 'float32');
    return imageTensor.div(255.0).expandDims(0);  // Normalize and expand dimensions
}

async function classifyImages(model, imagePaths) {
    for (const imagePath of imagePaths) {
        try {
            const inputTensor = await preprocessImage(imagePath);
            const predictions = model.predict(inputTensor);
            const predictionData = predictions.dataSync();

            const predictedClass = predictionData.indexOf(Math.max(...predictionData));
            const confidence = predictionData[predictedClass];

            displayResults(imagePath, predictedClass, confidence);
        } catch (error) {
            console.error(`Error processing image ${imagePath}:`, error.message);
        }
    }
}

function displayResults(imagePath, predictedClass, confidence) {
    console.log('----------------------------------------');
    console.log(`Image Path:       ${imagePath}`);
    console.log(`Predicted Class:  ${predictedClass}`);
    console.log(`Confidence:       ${confidence.toFixed(4)}`);
    console.log('----------------------------------------');
}

(async () => {
    if (process.argv.length < 3) {
        console.error('Usage: node cnn_app.js <image_path1> [<image_path2> ...]');
        process.exit(1);
    }

    const imagePaths = process.argv.slice(2);
    const model = await buildCNNModel();

    await classifyImages(model, imagePaths);
})();
