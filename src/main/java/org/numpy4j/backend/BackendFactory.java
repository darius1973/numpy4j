package org.numpy4j.backend;

import org.numpy4j.device.Device;
import org.numpy4j.device.DeviceType;

/**
 * Factory for obtaining the numerical backend associated with a device.
 *
 * <p>The factory provides a single shared instance of each supported
 * backend. Currently, only the CPU backend is implemented.</p>
 *
 * <p>Use this factory when explicitly selecting a backend for
 * device-specific array operations.</p>
 *
 * <pre>{@code
 * ArrayBackend backend = BackendFactory.get(Device.cpu());
 * NDArray result = backend.add(a, b);
 * }</pre>
 */
public final class BackendFactory {

    private static final ArrayBackend CPU =
            new CpuBackend();

    private BackendFactory() {
    }

    public static ArrayBackend get(Device device) {
        if (device.getType() == DeviceType.CPU) {
            return CPU;
        }

        throw new UnsupportedOperationException(
                "No backend implemented for " + device
        );
    }
}
