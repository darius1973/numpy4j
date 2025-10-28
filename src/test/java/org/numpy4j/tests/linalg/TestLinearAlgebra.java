package org.numpy4j.tests.linalg;

import org.junit.jupiter.api.Test;
import org.numpy4j.core.NDArray;
import org.numpy4j.linalg.LinearAlgebra;

import static org.junit.jupiter.api.Assertions.*;


public class TestLinearAlgebra {

    @Test
    public void testDot1D() {
        NDArray a = new NDArray(new double[]{1, 2, 3}, 3);
        NDArray b = new NDArray(new double[]{4, 5, 6}, 3);
        double result = LinearAlgebra.dot(a, b);
        assertEquals(32.0, result, 1e-10);
    }

    @Test
    public void testDotShapeMismatch() {
        NDArray a = new NDArray(new double[]{1, 2, 3}, 3);
        NDArray b = new NDArray(new double[]{1, 2, 3, 4}, 4);
        assertThrows(IllegalArgumentException.class, () -> LinearAlgebra.dot(a, b));
    }

    @Test
    public void testTranspose() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray t = LinearAlgebra.transpose(a);
        assertArrayEquals(new int[]{3, 2}, t.getShape());
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, t.getData(), 1e-10);
    }

    @Test
    public void testMatmul() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray b = new NDArray(new double[]{7, 8, 9, 10, 11, 12}, 3, 2);
        NDArray result = LinearAlgebra.matmul(a, b);
        assertArrayEquals(new int[]{2, 2}, result.getShape());
        assertArrayEquals(new double[]{58, 64, 139, 154}, result.getData(), 1e-10);
    }

    @Test
    public void testEye() {
        NDArray eye = LinearAlgebra.eye(3);
        assertArrayEquals(new int[]{3, 3}, eye.getShape());
        double[] expected = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
        };
        assertArrayEquals(expected, eye.getData(), 1e-10);
    }
}

