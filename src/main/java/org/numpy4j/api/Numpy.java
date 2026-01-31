package org.numpy4j.api;

import org.numpy4j.core.NDArray;

import java.util.ArrayList;
import java.util.Arrays;
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
     * Generate an NDArray of random values uniformly distributed in [0, 1).
     * Equivalent to Python's {@code numpy.random.random(size)} or {@code numpy.random.rand(*shape)}.
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray r = Numpy.random(2, 3);
     * System.out.println(r);
     * // [[0.7276, 0.6832, 0.3087]
     * //  [0.2771, 0.6655, 0.9033]]
     * }</pre>
     *
     * @param shape dimensions of the random array
     * @return NDArray filled with random doubles in [0, 1)
     */
    public static NDArray random(int... shape) {
        int size = 1;
        for (int s : shape) {
            size *= s;
        }

        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = RAND.nextDouble();
        }

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
     * Creates a new {@link NDArray} of the specified shape filled with ones.
     * <p>
     * This method mimics the behavior of <code>numpy.ones()</code> in Python.
     * It allocates an array of the given shape and initializes all elements to {@code 1.0}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray z = NDArray.ones(2, 3);
     * System.out.println(Arrays.toString(z.getData()));
     * // Output: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
     * }</pre>
     *
     * @param shape the dimensions of the array (e.g., {@code (2, 3)} for a 2×3 array)
     * @return a new {@link NDArray} with all elements initialized to one
     * @throws IllegalArgumentException if any dimension is non-positive
     */
    public static NDArray ones(int... shape) {
        int size = 1;
        for (int s : shape) size *= s;
        double[] data = new double[size];
        Arrays.fill(data, 1.0);
        return new NDArray(data, shape);
    }


    /**
     * Creates a new {@link NDArray} of the specified shape filled with a given constant value.
     * <p>
     * This method mimics the behavior of <code>numpy.full()</code> in Python.
     * It allocates an array of the given shape and fills all elements with {@code fillValue}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray f = Numpy.full(7.5, 2, 2);
     * System.out.println(Arrays.toString(f.getData()));
     * // Output: [7.5, 7.5, 7.5, 7.5]
     * }</pre>
     *
     * @param fillValue the value to fill the array with
     * @param shape the dimensions of the array
     * @return a new {@link NDArray} with all elements initialized to {@code fillValue}
     * @throws IllegalArgumentException if any dimension is non-positive
     */
    public static NDArray full(double fillValue, int... shape) {
        int size = 1;
        for (int s : shape) {
            if (s <= 0) throw new IllegalArgumentException("Shape dimensions must be positive");
            size *= s;
        }
        double[] data = new double[size];
        Arrays.fill(data, fillValue);
        return new NDArray(data, shape);
    }

    /**
     * Generates a one-dimensional {@link NDArray} of evenly spaced numbers over a specified interval.
     * <p>
     * This method mimics the behavior of <code>numpy.linspace()</code> in Python.
     * It generates {@code num} values starting at {@code start} and ending at {@code stop}, inclusive.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray a = Numpy.linspace(0, 1, 5);
     * System.out.println(Arrays.toString(a.getData()));
     * // Output: [0.0, 0.25, 0.5, 0.75, 1.0]
     * }</pre>
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of evenly spaced samples to generate
     * @return a new one-dimensional {@link NDArray} containing the generated values
     * @throws IllegalArgumentException if {@code num} is not positive
     */
    public static NDArray linspace(double start, double stop, int num) {
        if (num <= 0) throw new IllegalArgumentException("num must be > 0");
        double[] data = new double[num];
        if (num == 1) {
            data[0] = start;
        } else {
            double step = (stop - start) / (num - 1);
            for (int i = 0; i < num; i++) {
                data[i] = start + i * step;
            }
        }
        return new NDArray(data, num);
    }

    /**
     * Creates a two-dimensional identity matrix of size {@code n × n}.
     * <p>
     * This method mimics the behavior of <code>numpy.eye()</code> in Python.
     * The diagonal elements are set to {@code 1.0}, and all others are {@code 0.0}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray id = Numpy.eye(3);
     * System.out.println(Arrays.toString(id.getData()));
     * // Output: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
     * }</pre>
     *
     * @param n the number of rows and columns
     * @return a new {@link NDArray} representing the identity matrix
     * @throws IllegalArgumentException if {@code n} is non-positive
     */
    public static NDArray eye(int n) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0;
        }
        return new NDArray(data, n, n);
    }

    /**
     * Creates a deep copy of the given {@link NDArray}.
     * <p>
     * This method mimics the behavior of <code>numpy.copy()</code> in Python.
     * The returned array contains the same data but in a separate memory location.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray a = Numpy.ones(2,2);
     * NDArray b = Numpy.copy(a);
     * b.getData()[0] = 99;
     * // a remains unchanged
     * }</pre>
     *
     * @param a the input NDArray to copy
     * @return a new {@link NDArray} with copied data
     */
    public static NDArray copy(NDArray a) {
        double[] newData = Arrays.copyOf(a.getData(), a.getData().length);
        return new NDArray(newData, a.getShape());
    }

    /**
     * Flattens the input {@link NDArray} to a one-dimensional array.
     * <p>
     * This method mimics the behavior of <code>numpy.flatten()</code> in Python.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray a = Numpy.arange(1,7);
     * NDArray flat = Numpy.flatten(a);
     * System.out.println(Arrays.toString(flat.getData()));
     * // Output: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * }</pre>
     *
     * @param a the input NDArray
     * @return a new one-dimensional {@link NDArray} containing the same data
     */
    public static NDArray flatten(NDArray a) {
        double[] data = Arrays.copyOf(a.getData(), a.getData().length);
        return new NDArray(data, data.length);
    }

    /**
     * Alias for {@link #flatten(NDArray)}.
     * <p>
     * This method mimics the behavior of <code>numpy.ravel()</code> in Python.
     * </p>
     */
    public static NDArray ravel(NDArray a) {
        return flatten(a);
    }

    /**
     * Reshapes the input {@link NDArray} to the specified new shape.
     * <p>
     * This method mimics the behavior of <code>numpy.reshape()</code> in Python.
     * The total number of elements must remain the same.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray a = Numpy.arange(1,7); // 6 elements
     * NDArray reshaped = Numpy.reshape(a, 2,3);
     * System.out.println(Arrays.toString(reshaped.getData()));
     * // Output: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     * }</pre>
     *
     * @param a the input NDArray
     * @param newShape the desired shape
     * @return a new {@link NDArray} reshaped to {@code newShape}
     * @throws IllegalArgumentException if total size mismatches
     */
    public static NDArray reshape(NDArray a, int... newShape) {
        int newSize = 1;
        for (int s : newShape) newSize *= s;
        if (newSize != a.getData().length) {
            throw new IllegalArgumentException(
                    "Cannot reshape array of size " + a.getData().length + " into shape " + Arrays.toString(newShape));
        }
        return new NDArray(a.getData(), newShape);
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
     * Element-wise exponential of all elements in the input NDArray.
     * <p>
     * Equivalent to Python's {@code numpy.exp(x)}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray x = Numpy.arange(0, 5);
     * NDArray y = Numpy.exp(x);
     * System.out.println(y);
     * // Output: [1.0000, 2.7183, 7.3891, 20.0855, 54.5981]
     * }</pre>
     *
     * @param a input NDArray
     * @return a new NDArray with element-wise exponentials
     */
    public static NDArray exp(NDArray a) {
        double[] input = a.getData();
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i]);
        }
        return new NDArray(result, a.getShape());
    }

    /**
     * Compute the mean (average) of all elements in the input NDArray.
     * <p>
     * Equivalent to Python's {@code numpy.mean(x)}.
     * </p>
     *
     * <p><strong>Example usage:</strong></p>
     * <pre>{@code
     * NDArray x = Numpy.arange(1, 6);
     * double mean = Numpy.mean(x);
     * System.out.println(mean);
     * // Output: 3.0
     * }</pre>
     *
     * @param a input NDArray
     * @return mean value as a scalar double
     */
    public static double mean(NDArray a) {
        return a.mean();
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
