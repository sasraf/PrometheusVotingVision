package metalearnerNN.Activations;

import metalearnerNN.Matrix;

public class reluActivationFunction implements ActivationFunction {
    @Override
    public double[] activation(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Double.max(0, x[i]);
        }
        return x;
    }

    private double[] activationDerivative(double[] x) {
        for (int i = 0; i < x.length; i++) {
            if (x[i] <= 0) {
                x[i] = 0;
            } else {
                x[i] = 1;
            }
        }
        return x;
    }

    @Override
    public double[] dEdX(double[] input, double[] outputError) {
        return Matrix.matrixMultiply(activationDerivative(input), outputError);
    }
}
