package org.numpy4j.device;

import org.numpy4j.storage.ArrayStorage;
import org.numpy4j.storage.CpuStorage;

import java.util.Collections;
import java.util.List;

public final class DeviceManager {

    /**
     * Manages device availability and storage creation for numpy4j.
     *
     * <p>This class provides information about the devices supported
     * by the current implementation and creates the appropriate
     * {@link ArrayStorage} implementation for a selected device.</p>
     *
     * <p>Currently, only the CPU device is supported. CUDA device
     * discovery and storage creation may be added in future versions.</p>
     *
     * <p>This class is a utility class and cannot be instantiated.</p>
     */
    private DeviceManager() {
    }

    /**
     * Returns the devices currently supported by this implementation.
     *
     * @return an immutable list of available devices
     */
    public static List<Device> getAvailableDevices() {
        return Collections.singletonList(Device.cpu());
    }

    /**
     * Checks whether the specified device is available.
     *
     * @param device the device to check
     * @return true if the device is available
     */
    public static boolean isAvailable(Device device) {
        if (device == null) {
            return false;
        }

        if (device.getType() == DeviceType.CPU) {
            return device.getIndex() == 0;
        }

        // CUDA discovery will be implemented in a future backend.
        return false;
    }

    /**
     * Returns the default device.
     *
     * @return the CPU device
     */
    public static Device getDefaultDevice() {
        return Device.cpu();
    }

    public static ArrayStorage createStorage(Device device, int size) {
        if (!isAvailable(device)) {
            throw new UnsupportedOperationException(
                    "Device is unavailable: " + device
            );
        }

        if (device.equals(Device.cpu())) {
            return new CpuStorage(size);
        }

        throw new UnsupportedOperationException(
                "Storage implementation is not available for: " + device
        );
    }
}
