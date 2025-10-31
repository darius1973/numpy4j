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
    public void testReshapeValid() {
        NDArray a = new NDArray(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray reshaped = a.reshape(3, 2);

        // Shape should now be (3, 2)
        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());

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
}
