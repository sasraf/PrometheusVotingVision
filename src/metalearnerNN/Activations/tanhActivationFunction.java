package metalearnerNN.Activations;

public class tanhActivationFunction implements ActivationFunction {

    public tanhActivationFunction() {

    }

    //tanh activation function
    public double[] activation(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.tanh(x[i]);
        }
        return x;
    }

    // 1 - tanh(x)^2
    public double[] activationDerivative(double[] x) {
        x = activation(x);
        for (int i = 0; i < x.length; i++) {
            x[i] = 1 - Math.pow(x[i], 2);
        }

        return x;
    }

}
