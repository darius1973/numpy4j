package org.numpy4j.storage;

import java.util.Objects;

import org.numpy4j.device.Device;

/**
 * CPU-based implementation of {@link ArrayStorage}.
 *
 * <p>Stores array elements in a Java heap-allocated {@code double[]}.
 * All data is held in host memory and can be accessed directly through
 * indexed read and write operations.</p>
 *
 * <p>Input and output arrays are copied when transferring data to or
 * from this storage, preventing external modifications from affecting
 * the internal storage contents.</p>
 *
 * <p>Once {@link #close()} is called, further access to the storage
 * is prohibited and results in an {@link IllegalStateException}.</p>
 *
 * <p>Since this implementation uses Java-managed heap memory,
 * closing the storage does not require explicit memory deallocation.</p>
 */

public final class CpuStorage implements ArrayStorage {

    private final double[] data;
    private boolean closed;

    /**
     * Creates zero-initialized CPU storage.
     *
     * @param size number of elements
     */
    public CpuStorage(int size) {
        if (size < 0) {
            throw new IllegalArgumentException(
                    "Storage size cannot be negative"
            );
        }

        this.data = new double[size];
    }

    /**
     * Creates CPU storage by copying the supplied values.
     *
     * @param source initial values
     */
    public CpuStorage(double[] source) {
        Objects.requireNonNull(
                source, "Source data cannot be null"
        );

        this.data = source.clone();
    }

    @Override
    public int size() {
        ensureOpen();
        return data.length;
    }

    @Override
    public Device getDevice() {
        ensureOpen();
        return Device.cpu();
    }

    @Override
    public double[] copyToHost() {
        ensureOpen();
        return data.clone();
    }

    @Override
    public void copyFromHost(double[] source) {
        ensureOpen();

        Objects.requireNonNull(
                source, "Source data cannot be null"
        );

        if (source.length != data.length) {
            throw new IllegalArgumentException(
                    "Source length must match storage size"
            );
        }

        System.arraycopy(
                source, 0, data, 0, data.length
        );
    }

    @Override
    public void close() {
        // Java-owned heap memory needs no explicit release.
        // The closed flag prevents further access through this object.
        closed = true;
    }

    @Override
    public double get(int index) {
        ensureOpen();
        return data[index];
    }

    @Override
    public void set(int index, double value) {
        ensureOpen();
        data[index] = value;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "Storage has already been closed"
            );
        }
    }
}
