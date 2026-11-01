package org.numpy4j.storage;

import org.numpy4j.device.Device;

public interface ArrayStorage extends AutoCloseable {

    /**
     * Returns the number of stored elements.
     */
    int size();

    /**
     * Returns the device that owns this storage.
     */
    Device getDevice();

    /**
     * Copies the stored values into a host-side Java array.
     *
     * @return a copy of the stored values
     */
    double[] copyToHost();

    /**
     * Copies values from a host-side array into this storage.
     *
     * @param source the values to copy
     */
    void copyFromHost(double[] source);

    /**
     * Releases resources owned by this storage.
     */
    @Override
    void close();

    double get(int index);

    void set(int index, double value);
}
