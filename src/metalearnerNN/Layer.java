package metalearnerNN;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public interface Layer {

    public double[] feedForward(double[] layerInput);

    public double[] backProp(double[] outputError, double learningRate, double[] expected);

}

