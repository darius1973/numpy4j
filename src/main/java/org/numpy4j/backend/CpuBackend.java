package org.numpy4j.backend;

import java.util.Objects;

import org.numpy4j.core.NDArray;

public final class CpuBackend implements ArrayBackend {

    @Override
    public NDArray add(NDArray a, NDArray b) {
        validateInputs(a, b);
        return a.add(b);
    }

    @Override
    public NDArray subtract(NDArray a, NDArray b) {
        validateInputs(a, b);
        return a.subtract(b);
    }

    @Override
    public NDArray multiply(NDArray a, NDArray b) {
        validateInputs(a, b);
        return a.multiply(b);
    }

    @Override
    public NDArray divide(NDArray a, NDArray b) {
        validateInputs(a, b);
        return a.divide(b);
    }

    @Override
    public NDArray dot(NDArray a, NDArray b) {
        validateInputs(a, b);
        return a.dot(b);
    }

    private void validateInputs(NDArray a, NDArray b) {
        Objects.requireNonNull(a, "First array cannot be null");
        Objects.requireNonNull(b, "Second array cannot be null");
    }
}
