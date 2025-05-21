package org.numjava.linalg;

import org.numjava.core.NDArray;

public class LinearAlgebra {
    public static double dot(NDArray a, NDArray b) {
        double[] ad = a.getData();
        double[] bd = b.getData();
        if (ad.length != bd.length)
            throw new IllegalArgumentException("Arrays must be the same length for dot product");
        double sum = 0;
        for (int i = 0; i < ad.length; i++)
            sum += ad[i] * bd[i];
        return sum;
    }
}
