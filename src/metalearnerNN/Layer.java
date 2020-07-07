package metalearnerNN;

public interface Layer {

    public double[] feedForward(double[] layerInput);

    public double[] backProp(double[] outputError, double learningRate);

}

