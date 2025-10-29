package org.numpy4j.core;

import java.util.Arrays;

public class NDArray {
    private final int[] shape;
    private final int size;
    private final double[] data;

    public NDArray(int... shape) {
        this.shape = shape.clone();
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        this.data = new double[this.size];
    }

    public NDArray(double[] data, int... shape) {
        this.shape = shape.clone();
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (data.length != this.size)
            throw new IllegalArgumentException("Data size does not match shape");
        this.data = data.clone();
    }

    public double get(int... indices) {
        int idx = linearIndex(indices);
        return data[idx];
    }

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

    public NDArray add(NDArray other) {
        int[] resultShape = broadcastShape(this.shape, other.shape);
        NDArray result = new NDArray(resultShape);

        int[] idx = new int[resultShape.length];
        for (int i = 0; i < result.size; i++) {
            indexFromLinear(i, resultShape, idx);
            int thisIdx = linearIndexWithBroadcast(idx, this.shape);
            int otherIdx = linearIndexWithBroadcast(idx, other.shape);
            result.data[i] = this.data[thisIdx] + other.data[otherIdx];
        }
        return result;
    }

    private void checkShapeMatch(NDArray other) {
        if (!Arrays.equals(this.shape, other.shape))
            throw new IllegalArgumentException("Shapes do not match");
    }

    public int[] getShape() {
        return shape.clone();
    }

    public double[] getData() {
        return data.clone();
    }

    public NDArray reshape(int... newShape) {
        int newSize = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (newSize != size)
            throw new IllegalArgumentException("Total size must remain unchanged");
        return new NDArray(data, newShape);
    }

    // -------------------
    // New Methods
    // -------------------

    public double sum() {
        double s = 0;
        for (double v : data) s += v;
        return s;
    }

    public double mean() {
        return sum() / size;
    }

    public NDArray power(int exponent) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.pow(data[i], exponent);
        }
        return new NDArray(result, shape);
    }

    public NDArray transpose() {
        if (shape.length != 2) throw new UnsupportedOperationException("Only 2D transpose supported");
        int rows = shape[0], cols = shape[1];
        double[] result = new double[size];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[c * rows + r] = get(r, c);
            }
        }
        return new NDArray(result, cols, rows);
    }

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

    public NDArray dot(NDArray other) {
        // Only support 2D matrix multiplication
        if (this.shape.length != 2 || other.shape.length != 2)
            throw new UnsupportedOperationException("dot() currently only supports 2D matrices");

        int m = this.shape[0];
        int n = this.shape[1];
        int p = other.shape[1];

        if (n != other.shape[0])
            throw new IllegalArgumentException("Inner dimensions must match for dot product");

        double[] result = new double[m * p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                result[i * p + j] = sum;
            }
        }

        return new NDArray(result, m, p);
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

} 
