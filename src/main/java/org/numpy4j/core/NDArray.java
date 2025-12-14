package org.numpy4j.core;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.DoubleUnaryOperator;

/**
 * A lightweight, high-performance Java implementation of a multidimensional array,
 * inspired by Python's <strong>NumPy ndarray</strong>.
 * <p>
 * {@code NDArray} provides efficient storage and operations for numerical computing,
 * including element-wise arithmetic, slicing, reshaping, matrix multiplication,
 * broadcasting, and statistical methods.
 * <br>
 * It is designed as part of the {@code numpy4j} project — a Java library
 * bringing the expressiveness of NumPy to the JVM ecosystem.
 * </p>
 *
 * <h2>Key Features</h2>
 * <ul>
 *   <li>Support for arbitrary-dimensional arrays</li>
 *   <li>Element-wise arithmetic operations</li>
 *   <li>Automatic broadcasting (similar to NumPy)</li>
 *   <li>Reshape, slice, transpose, and power operations</li>
 *   <li>Matrix multiplication via {@link #dot(NDArray)}</li>
 *   <li>Statistical methods such as {@link #sum()} and {@link #mean()}</li>
 * </ul>
 *
 * <p><b>Example:</b></p>
 * <pre>{@code
 * // Create arrays
 * NDArray a = NDArray.ones(2, 3);
 * NDArray b = NDArray.arange(0, 6, 1).reshape(2, 3);
 *
 * // Perform operations
 * NDArray c = a.add(b);
 * NDArray d = c.power(2);
 * NDArray e = d.dot(NDArray.eye(3));
 *
 * System.out.println("Sum: " + e.sum());
 * System.out.println("Mean: " + e.mean());
 * }</pre>
 *
 * <p>
 * The internal data is stored as a flat {@code double[]} for efficiency,
 * and all operations are implemented in pure Java with optional multithreading.
 * Future versions may integrate with OpenBLAS or Intel MKL for native performance.
 * </p>
 *
 * @author
 *   Darius Nica
 * @version 1.0
 * @since 2025
 */
public class NDArray {
    private final int[] shape;
    private final int size;
    private final double[] data;

    // Concurrency
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();

    public NDArray(int... shape) {
        if (shape == null || shape.length == 0) throw new IllegalArgumentException("Shape cannot be null or empty");
        this.shape = shape.clone();
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        this.data = new double[this.size];
    }


    /**
     * Creates a new {@code NDArray} with the same structural metadata
     * (shape, layout) as this array, backed by the provided data buffer.
     * <p>
     * This method does <strong>not</strong> modify the current array.
     * The returned array is independent and does not share memory
     * with this instance.
     *
     * <p>Contract</p>
     * <ul>
     *   <li>The returned NDArray has the same shape as this array.</li>
     *   <li>The provided data buffer length must match actual data length.</li>
     *   <li>No data is copied from this array.</li>
     *   <li>The returned array does not share its backing buffer with this array.</li>
     * </ul>
     *
     * <p>Typical usage</p>
     * This method is commonly used by element-wise operations and
     * transformations that preserve shape:
     *
     * <pre>{@code
     * NDArray y = x.like(resultData);
     * }</pre>
     *
     * @param newData the backing data for the new NDArray
     * @return a new NDArray with identical shape and the given data
     * @throws IllegalArgumentException if {@code newData.length != size()}
     */
    public NDArray like(double[] newData) {
        if (newData.length != data.length) {
            throw new IllegalArgumentException(
                    "Data length " + newData.length + " does not match NDArray size " + data.length
            );
        }
        return new NDArray(newData, shape);
    }



    public NDArray(double[] data, int... shape) {
        this.shape = shape.clone();
        //this.strides = computeStrides(shape);
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (data.length != this.size)
            throw new IllegalArgumentException("Data size does not match shape");
        this.data = data.clone();
    }

    /**
     * Returns the element at the specified indices.
     *
     * @param indices the indices for each dimension
     * @return the element value
     * @throws IllegalArgumentException if index count or values are invalid
     */
    public double get(int... indices) {
        int idx = linearIndex(indices);
        return data[idx];
    }

    /**
     * Sets the element at the specified indices to the given value.
     *
     * @param value   the value to set
     * @param indices the indices for each dimension
     * @throws IllegalArgumentException if index count or values are invalid
     */
    public void set(double value, int... indices) {
        int idx = linearIndex(indices);
        data[idx] = value;
    }

    private int linearIndex(int... indices) {
        if (indices.length != shape.length)
            throw new IllegalArgumentException("Incorrect number of indices");
        int idx = 0, stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return idx;
    }

    /**
     * Adds this NDArray to another, applying NumPy-style broadcasting if necessary.
     *
     * @param other the other NDArray
     * @return a new NDArray containing the element-wise sum
     * @throws IllegalArgumentException if shapes are not broadcast-compatible
     */
    public NDArray add(NDArray other) {
        int[] outShape = broadcastShape(this.shape, other.shape);
        NDArray out = new NDArray(outShape);
        if (Arrays.equals(this.shape, other.shape)) {
            // parallel when large
            final int n = this.size;
            final int PAR_THRESHOLD = 1 << 14;
            if (n >= PAR_THRESHOLD) {
                int threads = Math.max(1, POOL.getParallelism());
                int block = (n + threads - 1) / threads;
                try {
                    POOL.submit(() -> {
                        for (int t = 0; t < threads; t++) {
                            final int s = t * block;
                            final int e = Math.min(n, s + block);
                            for (int i = s; i < e; i++) out.data[i] = this.data[i] + other.data[i];
                        }
                    }).get();
                } catch (Exception e) {
                    for (int i = 0; i < n; i++) out.data[i] = this.data[i] + other.data[i];
                }
            } else {
                for (int i = 0; i < n; i++) out.data[i] = this.data[i] + other.data[i];
            }
            return out;
        }
        // general broadcasting (serial)
        int[] idx = new int[outShape.length];
        for (int i = 0; i < out.size; i++) {
            indexFromLinear(i, outShape, idx);
            int ia = linearIndexWithBroadcast(idx, this.shape);
            int ib = linearIndexWithBroadcast(idx, other.shape);
            out.data[i] = this.data[ia] + other.data[ib];
        }
        return out;
    }

    /**
     * Applies the given unary function to each element of this NDArray.
     *
     * <p>This operation is element-wise and returns a new NDArray
     * with the same shape as the original.</p>
     *
     * <pre>{@code
     * NDArray x = NDArray.of(new double[]{-1, 2, -3, 4});
     * NDArray y = x.map(v -> Math.max(0, v)); // ReLU
     * }</pre>
     *
     * @param function a {@link DoubleUnaryOperator} representing the function to apply
     * @return a new {@code NDArray} with the result of applying {@code function} to each element
     */
    public NDArray map(DoubleUnaryOperator function) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = function.applyAsDouble(data[i]);
        }
        return new NDArray(result, shape.clone());
    }

    // Example helper to create an NDArray from a flat array and shape
    public static NDArray of(double[] values, int... shape) {
        return new NDArray(values, shape);
    }

    /**
     * Returns the shape (dimensions) of this NDArray.
     *
     * @return a copy of the shape array
     */
    public int[] getShape() {
        return shape.clone();
    }

    /**
     * Returns the data (dimensions) of this NDArray.
     *
     * @return a copy of the data array
     */
    public double[] getData() {
        return data.clone();
    }

    /**
     * Returns a new {@link NDArray} with the same data but a different shape.
     * <p>
     * This method behaves like <code>numpy.reshape()</code> in Python.
     * It does not modify the original array but creates a view-like copy
     * with the specified dimensions. The total number of elements must remain constant.
     * </p>
     *
     * <p><b>Example:</b></p>
     * <pre>{@code
     * NDArray a = NDArray.arange(0, 6, 1);  // shape (6,)
     * NDArray b = a.reshape(2, 3);          // shape (2, 3)
     *
     * System.out.println(Arrays.toString(a.getShape())); // [6]
     * System.out.println(Arrays.toString(b.getShape())); // [2, 3]
     * }</pre>
     *
     * @param newShape the target dimensions for the new array (e.g., {@code (2, 3)})
     * @return a reshaped {@link NDArray} containing the same data
     * @throws IllegalArgumentException if the total number of elements differs
     */
    public NDArray reshape(int... newShape) {
        int newSize = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (newSize != size)
            throw new IllegalArgumentException("Total size must remain unchanged");
        return new NDArray(data, newShape);
    }

    // -------------------
    // New Methods
    // -------------------
    /**
     * Computes the sum of all elements in the NDArray.
     *
     * @return the sum of all elements
     */
    public double sum() {
        double s = 0;
        for (double v : data) s += v;
        return s;
    }

    /**
     * Computes the mean (average) of all elements in the NDArray.
     *
     * @return the mean value
     */
    public double mean() {
        return sum() / size;
    }

    /**
     * Raises each element to the given power.
     *
     * @param exponent the exponent to apply
     * @return a new NDArray with each element raised to {@code exponent}
     */
    public NDArray power(int exponent) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.pow(data[i], exponent);
        }
        return new NDArray(result, shape);
    }

    /**
     * Extracts a subarray (slice) from this {@link NDArray}, similar to
     * <code>numpy[:, :]</code> slicing in Python.
     * <p>
     * Currently supports 2D arrays only. The slice indices are specified as
     * a 2D array of start and end indices per dimension.
     * </p>
     *
     * <p><b>Example:</b></p>
     * <pre>{@code
     * NDArray a = new NDArray(new double[]{
     *     1, 2, 3,
     *     4, 5, 6,
     *     7, 8, 9
     * }, 3, 3);
     *
     * // Slice rows 0–2 (exclusive of 2), columns 1–3 (exclusive of 3)
     * NDArray b = a.slice(new int[][]{{0, 2}, {1, 3}});
     *
     * // Result:
     * // [[2.0, 3.0],
     * //  [5.0, 6.0]]
     * }</pre>
     *
     * @param indices a 2D array defining start and end indices for each dimension,
     *                e.g. {@code {{rowStart, rowEnd}, {colStart, colEnd}}}
     * @return a new {@link NDArray} representing the requested slice
     * @throws IllegalArgumentException if the indices are invalid
     * @throws UnsupportedOperationException if the array is not 2D
     */
    public NDArray slice(int[][] indices) {
        if (indices.length != shape.length)
            throw new IllegalArgumentException("Slice indices must match number of dimensions");

        // Compute new shape
        int[] newShape = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            int start = indices[i][0];
            int end = indices[i][1];
            if (start < 0 || end > shape[i] || start >= end)
                throw new IllegalArgumentException("Invalid slice indices");
            newShape[i] = end - start;
        }

        // Simple approach: only support 2D slices for now
        if (shape.length == 2) {
            int rows = newShape[0], cols = newShape[1];
            double[] result = new double[rows * cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    result[r * cols + c] = get(r + indices[0][0], c + indices[1][0]);
                }
            }
            return new NDArray(result, newShape);
        } else {
            throw new UnsupportedOperationException("Slice only implemented for 2D arrays");
        }
    }

    /**
     * Computes the matrix dot product between this array and another.
     * <p>
     * This method is equivalent to <code>numpy.dot()</code> for 2D arrays (matrices).
     * It performs standard matrix multiplication: if {@code this} is of shape (m, n)
     * and {@code other} is of shape (n, p), the result will have shape (m, p).
     * </p>
     *
     * <p><b>Example:</b></p>
     * <pre>{@code
     * NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
     * NDArray b = new NDArray(new double[]{7, 8, 9, 10, 11, 12}, 3, 2);
     * NDArray c = a.dot(b);
     *
     * // Result:
     * // [[58.0, 64.0],
     * //  [139.0, 154.0]]
     *
     * System.out.println(Arrays.toString(c.getShape())); // [2, 2]
     * }</pre>
     *
     * @param other another {@link NDArray} to multiply with this one
     * @return the resulting {@link NDArray} after matrix multiplication
     * @throws UnsupportedOperationException if either array is not 2D
     * @throws IllegalArgumentException if the inner dimensions do not match
     */

    public NDArray dot(NDArray other) {
        if (this.shape.length != 2 || other.shape.length != 2)
            throw new UnsupportedOperationException("dot() supports 2D matrices only");
        int m = this.shape[0];
        int n = this.shape[1];
        int p = other.shape[1];
        if (n != other.shape[0]) throw new IllegalArgumentException("Inner dimensions must match");

        // if matrix is large, use parallel row-wise multiply
        final int PAR_MIN = 256 * 256; // number of elements threshold
        if ((long)m * p >= PAR_MIN) return javaDotParallel(other);
        return javaDot(other);
    }

    /*
      Single-threaded tiled matmul (cache-friendly)
      Uses cache-aware tiling
      Blocks fit in L1/L2 cache → minimal RAM access.
      Access patterns are sequential and predictable
      Uses the CPU prefetch efficiently
      Minimizes cache thrashing
      Ends up achieving ~20–30% of full BLAS performance
      For a pure-Java implementation with no native code and no vector API, this is very close to optimal.
      Tiling (also called blocking) means:
            Instead of multiplying whole matrices in one shot,
            break them into small square blocks (tiles) that fit in CPU cache.
      This means:
             - Process 64 rows of A at once
             - Multiply by 64 columns of B
             - Over a shared dimension chunk of 64
      Each 64×64 block is small enough to stay in L1 or L2 CPU cache, which gives a massive performance boost.
     */
    private NDArray javaDot(NDArray other) {
        int m = this.shape[0], n = this.shape[1], p = other.shape[1];
        double[] a = this.data, b = other.data;
        double[] c = new double[m * p];

        final int TILE_M = 64;
        final int TILE_N = 64;
        final int TILE_K = 64;

        for (int mm = 0; mm < m; mm += TILE_M) {
            int mLim = Math.min(mm + TILE_M, m);
            for (int kk = 0; kk < n; kk += TILE_K) {
                int kLim = Math.min(kk + TILE_K, n);
                for (int nn = 0; nn < p; nn += TILE_N) {
                    int nLim = Math.min(nn + TILE_N, p);
                    for (int i = mm; i < mLim; i++) {
                        int aRow = i * n;
                        int cRow = i * p;
                        for (int k = kk; k < kLim; k++) {
                            double av = a[aRow + k];
                            int bRow = k * p;
                            for (int j = nn; j < nLim; j++) {
                                c[cRow + j] += av * b[bRow + j];
                            }
                        }
                    }
                }
            }
        }
        return new NDArray(c, m, p);
    }

    // Parallel matmul: parallelize over output rows (simple, effective)
    private NDArray javaDotParallel(NDArray other) {
        int m = this.shape[0], n = this.shape[1], p = other.shape[1];
        double[] a = this.data, b = other.data;
        double[] c = new double[m * p];

        final int threads = Math.max(1, POOL.getParallelism());
        final int blockRows = (m + threads - 1) / threads;

        try {
            POOL.submit(() -> {
                for (int t = 0; t < threads; t++) {
                    final int rowStart = t * blockRows;
                    final int rowEnd = Math.min(m, rowStart + blockRows);
                    for (int i = rowStart; i < rowEnd; i++) {
                        int aRow = i * n;
                        int cRow = i * p;
                        for (int k = 0; k < n; k++) {
                            double av = a[aRow + k];
                            int bRow = k * p;
                            for (int j = 0; j < p; j++) {
                                c[cRow + j] += av * b[bRow + j];
                            }
                        }
                    }
                }
            }).get();
        } catch (Exception e) {
            // fallback to serial
            return javaDot(other);
        }

        return new NDArray(c, m, p);
    }


    /**
     * Creates an NDArray with the given shape, filled with random values
     * uniformly distributed between 0 (inclusive) and 1 (exclusive).
     *
     * <p>Example:
     * <pre>
     * NDArray arr = NDArrayUtils.random(2, 3);
     * // arr.shape() -> [2, 3]
     * // arr.data() contains 6 random double values
     * </pre>
     *
     * @param shape the shape of the NDArray, e.g., (2,3) for a 2x3 array
     * @return an NDArray of the specified shape with random values between 0 and 1
     * @throws IllegalArgumentException if any dimension is non-positive
     */
    public static NDArray random(int... shape) {
        if (shape.length == 0) throw new IllegalArgumentException("Shape must have at least one dimension");
        int size = 1;
        for (int s : shape) {
            if (s <= 0) throw new IllegalArgumentException("Shape dimensions must be positive");
            size *= s;
        }
        double[] data = new double[size];
        Random rand = new Random();
        for (int i = 0; i < size; i++) {
            data[i] = rand.nextDouble();
        }
        return new NDArray(data, shape);
    }

    /**
     * Creates an NDArray with the given shape, filled with zeros.
     *
     * <p>Example:
     * <pre>
     * NDArray arr = NDArrayUtils.zeros(2, 3);
     * // arr.shape() -> [2, 3]
     * // arr.data() contains 6 zeros
     * </pre>
     *
     * @param shape the shape of the NDArray, e.g., (2,3) for a 2x3 array
     * @return an NDArray of the specified shape filled with zeros
     * @throws IllegalArgumentException if any dimension is non-positive
     */
    public static NDArray zeros(int... shape) {
        if (shape.length == 0) throw new IllegalArgumentException("Shape must have at least one dimension");
        int size = 1;
        for (int s : shape) {
            if (s <= 0) throw new IllegalArgumentException("Shape dimensions must be positive");
            size *= s;
        }
        double[] data = new double[size]; // automatically initialized to 0
        return new NDArray(data, shape);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        formatArray(sb, data, shape, 0, 0);
        return sb.toString();
    }

    /**
     * Recursive formatter for multi-dimensional array printing.
     */
    private void formatArray(StringBuilder sb, double[] data, int[] shape, int dim, int offset) {
        int size = shape[dim];

        if (dim == shape.length - 1) {
            sb.append("[");
            for (int i = 0; i < size; i++) {
                sb.append(String.format("%.4f", data[offset + i]));
                if (i < size - 1) sb.append(", ");
            }
            sb.append("]");
        } else {
            sb.append("[");
            int stride = 1;
            for (int i = dim + 1; i < shape.length; i++) stride *= shape[i];
            for (int i = 0; i < size; i++) {
                if (i > 0) sb.append("\n").append(" ".repeat(2 * (dim + 1)));
                formatArray(sb, data, shape, dim + 1, offset + i * stride);
            }
            sb.append("]");
        }
    }


    // Computes the broadcasted shape
    private int[] broadcastShape(int[] a, int[] b) {
        int n = Math.max(a.length, b.length);
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            int aDim = i >= n - a.length ? a[i - (n - a.length)] : 1;
            int bDim = i >= n - b.length ? b[i - (n - b.length)] : 1;
            if (aDim != bDim && aDim != 1 && bDim != 1) throw new IllegalArgumentException("Shapes cannot be broadcast");
            result[i] = Math.max(aDim, bDim);
        }
        return result;
    }

    // Convert linear index in broadcasted array to original array index
    private int linearIndexWithBroadcast(int[] idx, int[] shape) {
        int offset = idx.length - shape.length;
        int linear = 0, stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            int dimIdx = idx[i + offset] % shape[i];
            linear += dimIdx * stride;
            stride *= shape[i];
        }
        return linear;
    }

    // Convert linear index to multidimensional index
    private void indexFromLinear(int linear, int[] shape, int[] idx) {
        for (int i = shape.length - 1; i >= 0; i--) {
            idx[i] = linear % shape[i];
            linear /= shape[i];
        }
    }

    /** Returns the number of elements in the array (like NumPy's ndarray.size) */
    public int getSize() {
        int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (this.size != data.length)
            throw new IllegalArgumentException("Data length does not match shape");
        return size;
    }

    /** Returns the number of dimensions (like NumPy's ndarray.ndim) */
    public int getNdims() {
        return shape.length;
    }

    // Returns a 1D view of row i (shape: [n])
    public double[] getRow(int r) {
        if (shape.length != 2) throw new UnsupportedOperationException("getRow only valid for 2D arrays");
        int m = shape[0], n = shape[1];
        if (r < 0 || r >= m) throw new IndexOutOfBoundsException();
        double[] row = new double[n];
        int offset = r * n;
        System.arraycopy(data, offset, row, 0, n);
        return row;
    }

}
