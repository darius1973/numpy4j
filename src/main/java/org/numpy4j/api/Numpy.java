package org.numjava.api;

import org.numjava.core.NDArray;

public class Numpy {
    public static NDArray array(double[] data, int... shape) {
        return new NDArray(data, shape);
    }

    public static NDArray zeros(int... shape) {
        return new NDArray(shape);
    }
}
