package org.numpy4j.activations;

import org.numpy4j.core.NDArray;

/**
 * Collection of common activation functions used in neural networks.
 * <p>
 * All activation functions operate element-wise on the input {@link NDArray}
 * and return a new {@code NDArray} with the same shape.
 * <p>
 * This class is stateless and cannot be instantiated.
 */
public final class Activations {

    /**
     * Prevents instantiation.
     */
    private Activations() {
    }

    /**
     * Applies the Rectified Linear Unit (ReLU) activation function.
     * <p>
     * {@code relu(x) = max(0, x)}
     * <p>
     * This operation is performed element-wise and preserves the shape
     * of the input array.
     *
     * @param x input NDArray
     * @return a new NDArray with ReLU applied
     */
    public static NDArray relu(NDArray x) {
        // Read-only access to input data
        double[] in = x.getData();
        // Allocate output buffer
        double[] out = new double[in.length];

        // Apply ReLU element-wise
        for (int i = 0; i < in.length; i++) {
            double v = in[i];
            out[i] = v > 0 ? v : 0.0;
        }

        // Preserve shape using structural cloning
        return x.like(out);
    }

    /**
     * Applies the sigmoid (logistic) activation function.
     * <p>
     * {@code sigmoid(x) = 1 / (1 + exp(-x))}
     * <p>
     * This function maps all values to the range (0, 1).
     *
     * @param x input NDArray
     * @return a new NDArray with sigmoid applied
     */
    public static NDArray sigmoid(NDArray x) {
        double[] in = x.getData();
        double[] out = new double[in.length];

        // Apply sigmoid element-wise
        for (int i = 0; i < in.length; i++) {
            out[i] = 1.0 / (1.0 + Math.exp(-in[i]));
        }

        return x.like(out);
    }

    /**
     * Applies the hyperbolic tangent (tanh) activation function.
     * <p>
     * {@code tanh(x)} maps values to the range (-1, 1).
     *
     * @param x input NDArray
     * @return a new NDArray with tanh applied
     */
    public static NDArray tanh(NDArray x) {
        double[] in = x.getData();
        double[] out = new double[in.length];

        // Apply tanh element-wise
        for (int i = 0; i < in.length; i++) {
            out[i] = Math.tanh(in[i]);
        }

        return x.like(out);
    }

    /**
     * Applies the softmax activation function.
     * <p>
     * Softmax converts a vector of values into a probability distribution
     * where all values are in the range (0, 1) and sum to 1.
     * <p>
     * This implementation uses the numerically stable form by subtracting
     * the maximum input value before exponentiation.
     *
     * @param x input NDArray (typically 1D)
     * @return a new NDArray representing a probability distribution
     */
    public static NDArray softmax(NDArray x) {
        double[] in = x.getData();
        double[] out = new double[in.length];

        // Find maximum value for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (double v : in) {
            max = Math.max(max, v);
        }

        // Compute exponentials and their sum
        double sum = 0.0;
        for (int i = 0; i < in.length; i++) {
            out[i] = Math.exp(in[i] - max);
            sum += out[i];
        }

        // Normalize to obtain probabilities
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }

        return x.like(out);
    }
}

