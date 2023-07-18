const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000 },
            options
        );

        this.m = 0;
        this.b = 0;
    }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent();
        }
    }

    gradientDescent() {
        const currentGuesses = this.features.map(row => {
            return this.m * row[0] + this.b;
        })

        const bSlope = _.sum(currentGuesses.map((guess, i) => {
            return guess - this.labels[i][0]
        })) * 2 / this.features.length;
    }
}

module.exports = LinearRegression;