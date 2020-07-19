package metalearnerNN;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public interface Layer {

    public double[] feedForward(double[] layerInput);


//TODO: REMOVE EXPECTED
    public double[] backProp(double[] outputError, double learningRate, double[] expected);

    public void save(FileWriter fileWriter) throws IOException;

}

