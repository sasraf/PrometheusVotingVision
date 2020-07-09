package metalearnerNN;

public interface ActivationFunction {

    public double[] activation(double[] x);

    public double[] activationDerivative(double[] x);
}
