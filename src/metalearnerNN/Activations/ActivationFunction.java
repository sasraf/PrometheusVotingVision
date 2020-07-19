package metalearnerNN.Activations;

public interface ActivationFunction {

    public double[] activation(double[] x);

//    public double[] activationDerivative(double[] x);
    public double[] dEdX(double[] input, double[] outputOroutputError);
}
