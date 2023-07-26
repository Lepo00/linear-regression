require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const _ = require('lodash');
const MultinominalLogisticRegression = require('./multinominal-logistic-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: 50,
    converters: {
        mpg: (value) => {
            const mpg = parseFloat(value);
            if (mpg < 15) {
                return [1, 0, 0]
            } else if (mpg < 30) {
                return [0, 1, 0]
            } else {
                return [0, 0, 1]
            }
        }
    }
});

const regression = new MultinominalLogisticRegression(features, _.flatMap(labels), {
    learningRate: .5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: .65
});

regression.train();
regression.predict([[130, 307, 1.75]]).print();
console.log(regression.test(testFeatures, _.flatMap(testLabels)));
