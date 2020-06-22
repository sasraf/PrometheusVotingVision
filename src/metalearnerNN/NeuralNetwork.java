package metalearnerNN;

import java.util.ArrayList;
import java.util.function.Function;

public class NeuralNetwork {

    ArrayList<Layer> layers;
    Function loss;
    Function lossDerivative;

    public NeuralNetwork() {
        layers = new ArrayList<Layer>();
        loss = null;
        lossDerivative = null;
    }

    // Adds layer
    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    // Sets loss function
    public  void setLoss(Function inputLoss, Function inputLossDerivative) {
        loss = inputLoss;
        lossDerivative = inputLossDerivative;
    }




}
