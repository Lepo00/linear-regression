const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class MultinominalLogisticRegression {
    constructor(features, labels, options) {
        this.features = this._processFeature(features);
        this.labels = tf.tensor(labels);
        this.costHistory = [];

        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000, batchSize: 10, decisionBoundary: 0.5 },
            options
        );

        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    }

    _processFeature(features) {
        features = tf.tensor(features);

        if (this.mean && this.variance) {
            features = features.sub(mean).div(variance.pow(.5));
        } else {
            features = this.standardize(features);
        }

        features = tf.ones(features.shape[0], 1).concat(features, 1);
        return features;
    }

    train() {
        const batchQty = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );

        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQty; j++) {
                const featureSlice = this.features.slice([j * this.options.batchSize, 0][this.options.batchSize, -1]);
                const labelSlice = this.labels.slice([j * this.options.batchSize, 0][this.options.batchSize, -1]);
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels).argMax(2);

        const incorrect = predictions
            .notEqual(testLabels)
            .sum()
            .get();

        const accuracy = (predictions.shape[0] - incorrect.get()) / predictions.shape[0];
        return accuracy;
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(.5));
    }

    recordCost() {
        const guesses = this.features.matMul(this.weights).softmax();

        const t1 = this.labels.transpose().matMul(guesses.log());
        const t2 = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(
                guesses
                    .mul(-1)
                    .add(1)
                    .log()
            )

        const cost = t1.add(t2).div(this.features.shape[0]).mul(-1).get(0, 0);
        this.costHistory.unshift(cost);
    }

    updateLearningRate() {
        if (this.costHistory.length > 2) return;

        const lastValue = this.costHistory[this.costHistory.length - 1];
        const secondLast = this.costHistory[this.costHistory.length - 2];

        if (lastValue > secondLast)
            this.options.learningRate /= 2;
        else
            this.options.learningRate *= 1.05;
    }

    predict(observations) {
        observations = this._processFeature(observations)
            .matMul(this.weights)
            .softmax()
            .argMax(1);
    }
}

module.exports = MultinominalLogisticRegression;