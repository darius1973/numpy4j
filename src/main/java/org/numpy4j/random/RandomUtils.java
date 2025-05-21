package org.numjava.random;

import java.util.Random;

public class RandomUtils {
    private static final Random RAND = new Random();

    public static double nextGaussian() {
        return RAND.nextGaussian();
    }

    public static double nextUniform(double min, double max) {
        return min + (max - min) * RAND.nextDouble();
    }
}
