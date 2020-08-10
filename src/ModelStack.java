import java.util.ArrayList;

public class ModelStack {

    // The number of algorithms and number of outputs per algorithm hardcoded
    private final int numOfPossibleOutputs = 4;
    private final int numOfAlgorithms = 4;

    private ArrayList<Algorithm> algorithms;

    //For testing purposes
    public ModelStack() {
        algorithms = new ArrayList<Algorithm>(numOfAlgorithms);

        // Stores all algorithms into an arrayList
        algorithms.add(new TestAlgo(new double[]{0, .1, .2, .3}));
        algorithms.add((new TestAlgo(new double[]{.4, .45, .5, .55})));
        algorithms.add(new TestAlgo(new double[]{.6, .65, .7, .75}));
        algorithms.add(new TestAlgo(new double[]{.8, .85, .9, .95}));

        // Loads training data into each algorithm
//        for (int i = 0; i < algorithms.size(); i++) {
//            algorithms.get(i).load();
//        }
    }

    // Takes in an already created ArrayList of algorithms
    public ModelStack(ArrayList<Algorithm> inputAlgorithms, ArrayList<String> savePaths) {
        algorithms = inputAlgorithms;

        // Loads training data into each algorithm
        for (int i = 0; i < algorithms.size(); i++) {
            algorithms.get(i).load(savePaths.get(i));
        }
    }

    public double[] processImage(String imagePath) {
        ArrayList<Double> outputArrayList = new ArrayList<Double>(numOfAlgorithms * numOfPossibleOutputs);

        // For each algorithm
        for (int i = 0; i < algorithms.size(); i++) {
            // Set the output of algorithm i to holder
            double[] holder = algorithms.get(i).processImage(imagePath);

            // For each value in holder
            for (int n = 0; n < holder.length; n++) {
                //Append value to output
                outputArrayList.add(holder[n]);
            }
        }

        // Convert arraylist to array for return (toArray creates an arrayList of Double, not double)
        double[] output = new double[outputArrayList.size()];

        for (int n = 0; n < output.length; n++)
        {
            output[n] = outputArrayList.get(n);
        }
        return output;
    }

    //TODO: maybe implement a train method that trains all the algorithms?
}
