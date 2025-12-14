package org.numpy4j.tests.core;

import org.junit.jupiter.api.Test;
import org.numpy4j.core.NDArray;

import static org.junit.jupiter.api.Assertions.*;

public class NDArrayTest {
    @Test
    public void testAdd() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray b = new NDArray(new double[]{6, 5, 4, 3, 2, 1}, 2, 3);
        NDArray c = a.add(b);
        assertArrayEquals(new double[]{7, 7, 7, 7, 7, 7}, c.getData(), 1e-10);
    }

    @Test
    public void testDot() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray b = new NDArray(new double[]{6, 5, 4, 3, 2, 1}, 3, 2);
        NDArray c = a.dot(b);
        assertArrayEquals(new double[]{20, 14, 56, 41}, c.getData(), 1e-10);
    }


    @Test
    public void testReshapeValid() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        System.out.println(a);
        NDArray reshaped = a.reshape(3, 2);

        // Shape should now be (3, 2)
        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());
        System.out.println(reshaped);
        // Data order must remain the same (row-major)
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, reshaped.getData(), 1e-10);
    }

    @Test
    public void testReshapeInvalid() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);

        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            a.reshape(4, 4); // total size changes from 6 → 16
        });

        assertTrue(exception.getMessage().contains("Total size must remain unchanged"));
    }

    @Test
    public void test2DArrayToString() {

        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        String expectedArrayFormat = "[[1,0000, 2,0000, 3,0000]\n" +
                "  [4,0000, 5,0000, 6,0000]]";

        assertEquals(a.toString(), expectedArrayFormat);
    }

    @Test
    public void test3DArrayToString() {

        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        String expectedArrayFormat = "[[1,0000, 2,0000, 3,0000]\n" +
                "  [4,0000, 5,0000, 6,0000]\n" +
                "  [7,0000, 8,0000, 9,0000]]";

        assertEquals(a.toString(), expectedArrayFormat);

    }

    @Test
    public void testGetSize() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(6, a.getSize());
    }

    @Test
    public void testGetNdims() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(2, a.getNdims());
    }

    @Test
    void testMapAppliesFunctionElementWise() {
        NDArray x = NDArray.of(new double[]{-1, 0, 2, -3}, 4);
        NDArray y = x.map(v -> Math.max(0, v)); // ReLU

        double[] expected = {0, 0, 2, 0};
        assertArrayEquals(expected, y.getData(), 1e-9, "ReLU should replace negatives with 0");
    }

    @Test
    void testMapPreservesShape() {
        NDArray x = NDArray.of(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray y = x.map(v -> v * 2);

        assertArrayEquals(new int[]{2, 3}, y.getShape());
        assertArrayEquals(new double[]{2, 4, 6, 8, 10, 12}, y.getData(), 1e-9);
    }

    @Test
    public void testRandomShapeAndValues() {
        NDArray arr = NDArray.random(2, 3);
        assertArrayEquals(new int[]{2, 3}, arr.getShape());
        for (double v : arr.getData()) {
            assertTrue(v >= 0.0 && v < 1.0, "Value must be in [0,1): " + v);
        }
    }

    @Test
    public void testZerosShapeAndValues() {
        NDArray arr = NDArray.zeros(2, 3);
        assertArrayEquals(new int[]{2, 3}, arr.getShape());
        for (double v : arr.getData()) {
            assertEquals(0.0, v, 1e-9);
        }
    }

    @Test
    public void testRandomIllegalShape() {
        assertThrows(IllegalArgumentException.class, NDArray::random);
        assertThrows(IllegalArgumentException.class, () -> NDArray.random(2, -3));
    }

    @Test
    public void testZerosIllegalShape() {
        assertThrows(IllegalArgumentException.class, NDArray::zeros);
        assertThrows(IllegalArgumentException.class, () -> NDArray.zeros(0, 3));
    }

    @Test
    public void testRandomDifferentCallsProduceDifferentArrays() {
        NDArray arr1 = NDArray.random(2, 2);
        NDArray arr2 = NDArray.random(2, 2);
        boolean equal = true;
        double[] data1 = arr1.getData();
        double[] data2 = arr2.getData();
        for (int i = 0; i < data1.length; i++) {
            if (data1[i] != data2[i]) {
                equal = false;
                break;
            }
        }
        assertFalse(equal, "Two random arrays should usually not be identical");
    }

    @Test
    void testLikePreserveShapeAndCopyDataReference() {
        NDArray original = new NDArray(
                new double[]{1.0, 2.0, 3.0, 4.0},
                2, 2
        );

        double[] newData = new double[]{10.0, 20.0, 30.0, 40.0};

        NDArray copy = original.like(newData);

        // shape must be identical
        assertArrayEquals(original.getShape(), copy.getShape());

        // data must contain the provided values
        assertArrayEquals(newData, copy.getData());

        // but must NOT share data with original
        assertNotSame(original.getData(), copy.getData());
    }

    @Test
    void testLikeSizeMismatchError() {
        NDArray original = new NDArray(
                new double[]{1.0, 2.0, 3.0, 4.0},
                2, 2
        );

        double[] wrongSize = new double[]{1.0, 2.0};

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> original.like(wrongSize)
        );

        assertTrue(ex.getMessage().toLowerCase().contains("size"));
    }

}
