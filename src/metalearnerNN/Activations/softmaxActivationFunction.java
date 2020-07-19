package metalearnerNN.Activations;

import metalearnerNN.Matrix;

public class softmaxActivationFunction implements ActivationFunction {

    private double esum;

    public double[] activation(double[] x) {
        //Get sum of each number taken to the power of e

        esum = 0;
        for (int i = 0; i < x.length; i++) {
            esum += Math.exp(x[i]);
        }

        // Run softmax function on each input in x
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i]) / esum;
        }

        return x;
    }

    public double[] dEdX(double[] input, double[] output) {
        double tSum = 0;
        for (int i = 0; i < output.length; i++) {
            tSum += output[i];
        }

        double[] o = activation(input);

        double[] c = new double[input.length];

        for (int i = 0; i < o.length; i++) {
            c[i] = o[i] * tSum - output[i];
        }
        return c;
    }
}
//
//    public double[] activationDerivativeWRTOutput
//}
