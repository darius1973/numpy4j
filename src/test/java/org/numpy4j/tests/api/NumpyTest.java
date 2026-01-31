package org.numpy4j.tests.api;


import org.junit.jupiter.api.Test;
import org.numpy4j.api.Numpy;
import org.numpy4j.core.NDArray;

import static org.junit.jupiter.api.Assertions.*;

public class NumpyTest {

    @Test
    public void testArange() {
        NDArray a = Numpy.arange(0, 5, 1);

        assertArrayEquals(new double[]{0, 1, 2, 3, 4}, a.getData(), 1e-10);
    }

    @Test
    public void testArangeDifferentStep() {
        NDArray a = Numpy.arange(0, 5, 2);

        assertArrayEquals(new double[]{0, 2, 4}, a.getData(), 1e-10);
    }

    @Test
    public void testArangeDifferentStop() {
        NDArray a = Numpy.arange(0, 10, 2);

        assertArrayEquals(new double[]{0, 2, 4, 6, 8}, a.getData(), 1e-10);
    }

    @Test
    public void testArrayCreation() {
        NDArray a = Numpy.array(new double[]{1, 2, 3, 4}, 2, 2);

        assertArrayEquals(new double[]{1, 2, 3, 4}, a.getData(), 1e-10);
        assertArrayEquals(new int[]{2, 2}, a.getShape());
    }

    @Test
    public void testZeros() {
        NDArray z = Numpy.zeros(3, 1);

        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, z.getData(), 1e-10);
        assertArrayEquals(new int[]{3, 1}, z.getShape());
    }

    @Test
    public void testEmptyDeterministic() {
        NDArray e1 = Numpy.zeros(42, 2, 2);
        NDArray e2 = Numpy.zeros(42, 2, 2);

        assertArrayEquals(e1.getData(), e2.getData(), 1e-10);
    }

    @Test
    public void testRandomShape() {
        NDArray r = Numpy.random(2, 3);
        assertEquals(6, r.getSize());
        assertArrayEquals(new int[]{2, 3}, r.getShape());
    }

    @Test
    public void testRandomReproducibility() {
        NDArray r1 = Numpy.random(2, 2);
        NDArray r2 = Numpy.random(2, 2);

        // Since RAND is deterministic but continuous, next calls will differ
        assertNotEquals(r1.getData()[0], r2.getData()[0]);
    }

    @Test
    public void testRandomRange() {
        NDArray r = Numpy.random(100);
        for (double v : r.getData()) {
            assertTrue(v >= 0.0 && v < 1.0, "Value out of [0,1): " + v);
        }
    }

    @Test
    public void testOnes() {
        NDArray a = Numpy.ones(2, 3);
        assertArrayEquals(new double[]{1,1,1,1,1,1}, a.getData());
        assertArrayEquals(new int[]{2,3}, a.getShape());
    }

    @Test
    public void testFull() {
        NDArray a = Numpy.full(7.5, 2, 2);
        assertArrayEquals(new double[]{7.5,7.5,7.5,7.5}, a.getData());
        assertArrayEquals(new int[]{2,2}, a.getShape());
    }

    @Test
    public void testLinspace() {
        NDArray a = Numpy.linspace(0, 1, 5);
        assertArrayEquals(new double[]{0.0,0.25,0.5,0.75,1.0}, a.getData(), 1e-10);
        assertArrayEquals(new int[]{5}, a.getShape());
    }

    @Test
    public void testEye() {
        NDArray a = Numpy.eye(3);
        assertArrayEquals(new double[]{
                1.0,0.0,0.0,
                0.0,1.0,0.0,
                0.0,0.0,1.0
        }, a.getData());
        assertArrayEquals(new int[]{3,3}, a.getShape());
    }

    @Test
    public void testCopy() {
        NDArray a = Numpy.ones(2,2);
        NDArray b = Numpy.copy(a);
        assertArrayEquals(a.getData(), b.getData());
        assertArrayEquals(a.getShape(), b.getShape());
    }

    @Test
    public void testFlattenRavel() {
        NDArray a = Numpy.arange(1, 7, 1); // [1,2,3,4,5,6]
        NDArray flat = Numpy.flatten(a);
        NDArray ravel = Numpy.ravel(a);
        assertArrayEquals(new double[]{1,2,3,4,5,6}, flat.getData());
        assertArrayEquals(new double[]{1,2,3,4,5,6}, ravel.getData());
    }

    @Test
    public void testReshape() {
        NDArray a = Numpy.arange(1,7,1); // 6 elements
        NDArray reshaped = Numpy.reshape(a, 2,3);
        assertArrayEquals(new int[]{2,3}, reshaped.getShape());
        assertArrayEquals(a.getData(), reshaped.getData());
    }
}
