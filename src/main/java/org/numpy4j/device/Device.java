package org.numpy4j.device;

import java.util.Objects;

/**
 * Represents a computing device available to numpy4j.
 *
 * <p>A device is identified by its {@link DeviceType} and an index.
 * The type specifies the device category (such as CPU or CUDA),
 * while the index identifies a specific device within that category.</p>
 *
 * <p>CPU devices use index {@code 0}. CUDA devices use an index
 * corresponding to the selected GPU.</p>
 *
 * <p>Device instances are immutable and can be compared using
 * {@link #equals(Object)}.</p>
 */
public final class Device {

    private final DeviceType type;
    private final int index;

    public Device(DeviceType type, int index) {
        this.type = Objects.requireNonNull(
                type, "Device type cannot be null"
        );

        if (index < 0) {
            throw new IllegalArgumentException(
                    "Device index cannot be negative"
            );
        }

        this.index = index;
    }

    public DeviceType getType() {
        return type;
    }

    public int getIndex() {
        return index;
    }

    public static Device cpu() {
        return new Device(DeviceType.CPU, 0);
    }

    public static Device cuda(int index) {
        return new Device(DeviceType.CUDA, index);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (!(obj instanceof Device)) {
            return false;
        }

        Device other = (Device) obj;

        return type == other.type && index == other.index;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, index);
    }

    @Override
    public String toString() {
        return type == DeviceType.CPU
                ? "CPU"
                : type + ":" + index;
    }
}
