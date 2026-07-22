package org.numpy4j.tests.linalg;

import org.junit.jupiter.api.Test;
import org.numpy4j.api.Numpy;
import org.numpy4j.core.NDArray;
import org.numpy4j.linalg.LinearAlgebra;

import static org.junit.jupiter.api.Assertions.*;


public class LinearAlgebraTest {

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
    void testRankFullRankMatrix() {
        NDArray A = NDArray.of(new double[]{
                1, 2,
                3, 4
        }, 2, 2);

        assertEquals(2, LinearAlgebra.rank(A));
    }

    @Test
    void testRankRankOneMatrix() {
        NDArray A = NDArray.of(new double[]{
                1, 2,
                2, 4
        }, 2, 2);

        assertEquals(1, LinearAlgebra.rank(A));
    }

    @Test
    void testRankZeroMatrix() {
        NDArray A = NDArray.zeros(3, 3);

        assertEquals(0, LinearAlgebra.rank(A));
    }

    @Test
    void testRankIdentityMatrix() {
        NDArray A = Numpy.eye(4);

        assertEquals(4, LinearAlgebra.rank(A));
    }

    @Test
    void testRankRectangularMatrix() {
        NDArray A = NDArray.of(new double[]{
                1, 2, 3,
                2, 4, 6
        }, 2, 3);

        assertEquals(1, LinearAlgebra.rank(A));
    }

    @Test
    void testRankRejectsNonMatrix() {
        NDArray A = NDArray.of(new double[]{1, 2, 3}, 3);

        assertThrows(IllegalArgumentException.class,
                () -> LinearAlgebra.rank(A));
    }

}

