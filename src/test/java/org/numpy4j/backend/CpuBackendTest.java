package org.numpy4j.backend;

import org.junit.jupiter.api.Test;
import org.numpy4j.core.NDArray;
import org.numpy4j.device.Device;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class CpuBackendTest {

    @Test
    public void testAdd() {
        NDArray a = NDArray.of(
                new double[]{1, 2, 3, 4}, 2, 2
        );

        NDArray b = NDArray.of(
                new double[]{5, 6, 7, 8}, 2, 2
        );

        ArrayBackend backend =
                BackendFactory.get(Device.cpu());

        NDArray result = backend.add(a, b);

        assertArrayEquals(
                new double[]{6, 8, 10, 12},
                result.getData(),
                1e-9
        );
    }

    @Test
    public void testMultiply() {
        NDArray a = NDArray.of(
                new double[]{1, 2, 3, 4}, 2, 2
        );

        NDArray b = NDArray.of(
                new double[]{5, 6, 7, 8}, 2, 2
        );

        NDArray result =
                BackendFactory.get(Device.cpu())
                        .multiply(a, b);

        assertArrayEquals(
                new double[]{5, 12, 21, 32},
                result.getData(),
                1e-9
        );
    }


    @Test
    public void testSubtract() {
        NDArray a = NDArray.of(
                new double[]{1, 2, 3, 4}, 2, 2
        );

        NDArray b = NDArray.of(
                new double[]{5, 6, 7, 8}, 2, 2
        );

        NDArray result =
                BackendFactory.get(Device.cpu())
                        .subtract(b, a);
        System.out.println(result.toString());
        assertArrayEquals(
                new double[]{4, 4, 4, 4},
                result.getData(),
                1e-9
        );
    }


    @Test
    public void testDivide() {
        NDArray a = NDArray.of(
                new double[]{1, 2, 3, 4}, 2, 2
        );

        NDArray b = NDArray.of(
                new double[]{4, 8, 12, 16}, 2, 2
        );

        NDArray result =
                BackendFactory.get(Device.cpu())
                        .divide(b, a);
        System.out.println(result.toString());
        assertArrayEquals(
                new double[]{4, 4, 4, 4},
                result.getData(),
                1e-9
        );
    }

}