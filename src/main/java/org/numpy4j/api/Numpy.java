package org.numpy4j.api;

import org.numpy4j.core.NDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Numpy {

    private static final Random RAND = new Random(42);


    /**
     * Creates a new {@link NDArray} from an existing one-dimensional Java array and a specified shape.
     * <p>
     * This method mimics the behavior of <code>numpy.array()</code> in Python.
     * It wraps a given {@code double[]} array into an {@link NDArray} of the specified dimensions.
     * </p>
     *
     * <p>
     * The total number of elements implied by the shape must match the length of the input array,
     * otherwise an {@link IllegalArgumentException} will be thrown.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * double[] data = {1.0, 2.0, 3.0, 4.0};
     * NDArray a = NDArray.array(data, 2, 2);
     * System.out.println(Arrays.toString(a.getData()));
     * // Output: [1.0, 2.0, 3.0, 4.0]
     * }</pre>
     *
     * @param data  the one-dimensional data array containing the elements
     * @param shape the desired dimensions of the resulting array (e.g., {@code (2, 2)})
     * @return a new {@link NDArray} wrapping the given data with the specified shape
     * @throws IllegalArgumentException if the data length does not match the product of the shape dimensions
     */
    public static NDArray array(double[] data, int... shape) {

        return new NDArray(data, shape);
    }

    /**
     * Creates a new {@link NDArray} of the specified shape filled with zeros.
     * <p>
     * This method mimics the behavior of <code>numpy.zeros()</code> in Python.
     * It allocates an array of the given shape and initializes all elements to {@code 0.0}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray z = NDArray.zeros(2, 3);
     * System.out.println(Arrays.toString(z.getData()));
     * // Output: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     * }</pre>
     *
     * @param shape the dimensions of the array (e.g., {@code (2, 3)} for a 2×3 array)
     * @return a new {@link NDArray} with all elements initialized to zero
     * @throws IllegalArgumentException if any dimension is non-positive
     */
    public static NDArray zeros(int... shape) {

        return new NDArray(shape);
    }

    /**
     * Creates a new {@link NDArray} of the specified shape with uninitialized (random) values.
     * <p>
     * This method mimics the behavior of <code>numpy.empty()</code> in Python.
     * It allocates an array of the given shape without setting the elements to zero,
     * which can be useful for performance when the contents will be overwritten later.
     * <br>
     * Since Java does not expose uninitialized memory, this implementation fills the array
     * with random values to simulate that behavior.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray a = NDArray.empty(2, 3);
     * System.out.println(Arrays.deepToString(a.reshape(2, 3).getData()));
     * // Output: [[0.384, 0.912, 0.156], [0.771, 0.428, 0.623]]  (values will vary)
     * }</pre>
     *
     * @param shape the dimensions of the array (e.g., {@code (2, 3)} for a 2×3 array)
     * @return a new {@link NDArray} with random values simulating uninitialized data
     * @throws IllegalArgumentException if any dimension is non-positive
     */
    public static NDArray empty(int... shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = RAND.nextDouble(); // simulate uninitialized memory
        }
        return new NDArray(data, shape);
    }

    /**
     * Creates a one-dimensional {@link NDArray} containing evenly spaced values within a specified interval.
     * <p>
     * This method mimics the behavior of <code>numpy.arange()</code> in Python.
     * It generates values starting from {@code start} (inclusive) up to {@code stop} (exclusive),
     * incremented by {@code step}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray a = NDArray.arange(0.0, 5.0, 1.0);
     * System.out.println(Arrays.toString(a.getData()));
     * // Output: [0.0, 1.0, 2.0, 3.0, 4.0]
     *
     * NDArray b = NDArray.arange(5.0, 0.0, -1.0);
     * System.out.println(Arrays.toString(b.getData()));
     * // Output: [5.0, 4.0, 3.0, 2.0, 1.0]
     * }</pre>
     *
     * @param start the starting value of the sequence (inclusive)
     * @param stop the end value of the sequence (exclusive)
     * @param step the spacing between values; must not be zero
     * @return a one-dimensional {@link NDArray} containing the generated values
     * @throws IllegalArgumentException if {@code step} is zero
     */
    public static NDArray arange(double start, double stop, double step) {
        if (step == 0) throw new IllegalArgumentException("Step cannot be zero");
        List<Double> values = new ArrayList<>();
        if (step > 0) {
            for (double val = start; val < stop; val += step) {
                values.add(val);
            }
        } else {
            for (double val = start; val > stop; val += step) {
                values.add(val);
            }
        }
        double[] data = values.stream().mapToDouble(Double::doubleValue).toArray();
        return new NDArray(data, data.length);
    }

}
