package org.numpy4j.api;

import org.numpy4j.core.NDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Numpy {

    private static final Random RAND = new Random(42);

    public static NDArray array(double[] data, int... shape) {
        return new NDArray(data, shape);
    }

    public static NDArray zeros(int... shape) {
        return new NDArray(shape);
    }

    public static NDArray empty(int... shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = RAND.nextDouble(); // simulate uninitialized memory
        }
        return new NDArray(data, shape);
    }

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
