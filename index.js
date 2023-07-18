require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    dataColumns: ['horsepower'],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: 50
});

