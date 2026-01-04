package org.numpy4j.ml;

import org.numpy4j.activations.Activations;
import org.numpy4j.core.NDArray;

/**
 * Binary Logistic Regression implemented using numpy4j NDArray.
 * <p>
 * This implementation is intentionally explicit and loop-based,
 * avoiding hidden abstractions or unsupported NDArray operations.
 */
public final class LogisticRegression {

    /** Weight vector of shape [numFeatures, 1] */
    private final NDArray weights;

    /** Bias term */
    private double bias;

    /**
     * Creates a Logistic Regression model.
     *
     * @param numFeatures number of input features
     */
    public LogisticRegression(int numFeatures) {
        this.weights = NDArray.zeros(numFeatures, 1);
        this.bias = 0.0;
    }

    /**
     * Computes predicted probabilities using the sigmoid function.
     *
     * @param x input matrix of shape [numSamples, numFeatures]
     * @return probabilities of shape [numSamples, 1]
     */
    public NDArray predictProba(NDArray x) {
        // z = X · w + b
        NDArray z = x.dot(weights);
        double[] zd = z.getData();

        for (int i = 0; i < zd.length; i++) {
            zd[i] += bias;
        }

        return Activations.sigmoid(z);
    }

    /**
     * Predicts binary class labels using a threshold of 0.5.
     *
     * @param x input features
     * @return predicted labels (0 or 1)
     */
    public NDArray predict(NDArray x) {
        NDArray probs = predictProba(x);
        double[] p = probs.getData();
        double[] out = new double[p.length];

        for (int i = 0; i < p.length; i++) {
            out[i] = p[i] >= 0.5 ? 1.0 : 0.0;
        }

        return probs.like(out);
    }

    /**
     * Trains the model using batch gradient descent.
     *
     * @param x input features [numSamples, numFeatures]
     * @param y target labels [numSamples, 1]
     * @param epochs number of training iterations
     * @param learningRate learning rate
     */
    public void fit(NDArray x, NDArray y, int epochs, double learningRate) {

        int numSamples = x.getShape()[0];
        int numFeatures = x.getShape()[1];

        double[] xData = x.getData();
        double[] yData = y.getData();
        double[] wData = weights.getData();

        for (int epoch = 0; epoch < epochs; epoch++) {

            // Forward pass
            NDArray yHat = predictProba(x);
            double[] yHatData = yHat.getData();

            // Gradient accumulators
            double[] gradW = new double[wData.length];
            double gradB = 0.0;

            // Compute gradients explicitly
            for (int i = 0; i < numSamples; i++) {
                double error = yHatData[i] - yData[i];
                gradB += error;

                for (int j = 0; j < numFeatures; j++) {
                    gradW[j] += xData[i * numFeatures + j] * error;
                }
            }

            // Average gradients
            gradB /= numSamples;
            for (int j = 0; j < gradW.length; j++) {
                gradW[j] /= numSamples;
            }

            // Update parameters
            for (int j = 0; j < wData.length; j++) {
                wData[j] -= learningRate * gradW[j];
            }
            bias -= learningRate * gradB;
        }
    }

    /**
     * Returns the learned weights.
     */
    public NDArray getWeights() {
        return weights;
    }

    /**
     * Returns the bias term.
     */
    public double getBias() {
        return bias;
    }
}
