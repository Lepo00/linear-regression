require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const mnist = require('mnist-data');
const MultinominalLogisticRegressionImage = require('./multinominal-logistic-regression-image');

function loadData() {
    const mnistData = mnist.training(0, 60000);
    const features = mnistData.images.values.map(image => _.flatMap(image));
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });
    return { features, labels: encodedLabels };
}

const { features, labels } = loadData();
const regression = new MultinominalLogisticRegressionImage(features, labels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100
});

regression.train();

const testingData = mnist.testing(0, 1000);
const testFeatures = testingData.images.values.map(image => _.flatMap(image));
const testEncodingLabels = testingData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
});
const accuracy = regression.test(testFeatures, testEncodingLabels);
console.log("Accuracy:" + accuracy);