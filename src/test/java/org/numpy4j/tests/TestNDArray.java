package org.numjava.tests;

import org.junit.jupiter.api.Test;
import org.numjava.core.NDArray;

import static org.junit.jupiter.api.Assertions.*;

public class TestNDArray {
    @Test
    public void testAdd() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray b = new NDArray(new double[]{6, 5, 4, 3, 2, 1}, 2, 3);
        NDArray c = a.add(b);
        assertArrayEquals(new double[]{7, 7, 7, 7, 7, 7}, c.getData(), 1e-10);
    }
}
