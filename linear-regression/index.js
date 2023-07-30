require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: 50
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.01,
    iterations: 100,
    batchSize: 10
});

regression.train();
regression.test();

