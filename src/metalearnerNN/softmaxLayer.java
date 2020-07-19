//package metalearnerNN;
//
//import java.io.FileWriter;
//import java.io.IOException;
//
//public class softmaxLayer implements Layer {
//
//
//    public double[] feedForward(double[] layerInput) {
//
//        double esum = 0;
//
//        for (int i = 0; i < layerInput.length; i++) {
//            esum += Math.exp(layerInput[i]);
//        }
//
//        // Run softmax function on each input in x
//        for (int i = 0; i < layerInput.length; i++) {
//            layerInput[i] = Math.exp(layerInput[i]) / esum;
//        }
//    }
//
//
//    public double[] backProp(double[] outputError, double learningRate) {
//        return new double[0];
//    }
//
//
//    public void save(FileWriter fileWriter) throws IOException {
//
//    }
//}
