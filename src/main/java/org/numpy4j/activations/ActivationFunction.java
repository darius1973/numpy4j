package org.numpy4j.activations;

import org.numpy4j.core.NDArray;

/**
 * Represents an activation function applied to an {@link NDArray}.
 * <p>
 * An activation function is a stateless, pure transformation that maps
 * an input NDArray to a new NDArray of the same shape.
 * <p>
 * This interface is a {@link FunctionalInterface} and is intended to be
 * used with lambdas or method references.
 *
 * <h2>Contract</h2>
 * <ul>
 *   <li>The implementation must not modify the input NDArray.</li>
 *   <li>The returned NDArray must have the same shape as the input.</li>
 *   <li>The implementation must be free of side effects.</li>
 * </ul>
 *
 * <h2>Typical usage</h2>
 * Standard activation functions can be referenced from
 * the {@link Activations} utility class:
 *
 * <pre>{@code
 * ActivationFunction relu = Activations::relu;
 * ActivationFunction sigmoid = Activations::sigmoid;
 * }</pre>
 *
 * Custom activation functions can be defined using lambdas:
 *
 * <pre>{@code
 * ActivationFunction leakyRelu = x -> {
 *     double[] in = x.getData();
 *     double[] out = new double[in.length];
 *
 *     for (int i = 0; i < in.length; i++) {
 *         double v = in[i];
 *         out[i] = v > 0 ? v : 0.01 * v;
 *     }
 *     return x.like(out);
 * };
 * }</pre>
 *
 * <p>
 * This interface is typically used as a strategy for neural network layers,
 * allowing activation behavior to be configured without inheritance or
 * conditional logic.
 */
@FunctionalInterface
public interface ActivationFunction {

    /**
     * Applies the activation function to the given NDArray.
     *
     * @param x input NDArray
     * @return a new NDArray with the activation function applied
     */
    NDArray apply(NDArray x);
}

