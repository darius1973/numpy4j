package org.numpy4j.core;

import java.util.Arrays;
import java.util.concurrent.*;

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
        checkShapeMatch(other);
        NDArray result = new NDArray(shape);
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        int chunk = size / Runtime.getRuntime().availableProcessors();
        for (int i = 0; i < size; i += chunk) {
            final int start = i;
            final int end = Math.min(size, i + chunk);
            executor.submit(() -> {
                for (int j = start; j < end; j++) {
                    result.data[j] = this.data[j] + other.data[j];
                }
            });
        }
        executor.shutdown();
        try { executor.awaitTermination(1, TimeUnit.MINUTES); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
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

    // Additional methods: slicing, broadcasting, etc. can be added similarly.
} 
