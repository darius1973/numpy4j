package org.numpy4j.backend;

import org.numpy4j.core.NDArray;

public interface ArrayBackend {

        NDArray add(NDArray a, NDArray b);

        NDArray subtract(NDArray a, NDArray b);

        NDArray multiply(NDArray a, NDArray b);

        NDArray divide(NDArray a, NDArray b);

        NDArray dot(NDArray a, NDArray b);

}
