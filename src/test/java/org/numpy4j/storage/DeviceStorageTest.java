package org.numpy4j.storage;

import org.numpy4j.device.Device;
import org.numpy4j.device.DeviceManager;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DeviceStorageTest {

    @Test
    public void testCpuDeviceAvailable() {
        assertTrue(
                DeviceManager.isAvailable(Device.cpu())
        );

        assertFalse(
                DeviceManager.isAvailable(Device.cuda(0))
        );
    }

    @Test
    public void testCpuStorageCopiesData() {
        double[] original = {1, 2, 3};

        CpuStorage storage = new CpuStorage(original);

        original[0] = 99;

        assertArrayEquals(
                new double[]{1, 2, 3},
                storage.copyToHost(),
                1e-9
        );

        double[] returned = storage.copyToHost();
        returned[1] = 99;

        assertArrayEquals(
                new double[]{1, 2, 3},
                storage.copyToHost(),
                1e-9
        );
    }

    @Test
    public void testCopyFromHost() {
        CpuStorage storage = new CpuStorage(3);

        storage.copyFromHost(new double[]{4, 5, 6});

        assertArrayEquals(
                new double[]{4, 5, 6},
                storage.copyToHost(),
                1e-9
        );
    }

    @Test
    public void testStorageRejectsWrongSize() {
        CpuStorage storage = new CpuStorage(3);

        assertThrows(
                IllegalArgumentException.class,
                () -> storage.copyFromHost(new double[]{1, 2})
        );
    }

    @Test
    public void testClosedStorageCannotBeUsed() {
        CpuStorage storage = new CpuStorage(3);

        storage.close();

        assertThrows(
                IllegalStateException.class,
                storage::copyToHost
        );
    }
}