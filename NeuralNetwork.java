import java.util.Random;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] weights1;
    private double[][] weights2;
    private double[] bias1;
    private double[] bias2;
    private double learningRate = 0.01;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        weights1 = new double[inputSize][hiddenSize];
        weights2 = new double[hiddenSize][outputSize];
        bias1 = new double[hiddenSize];
        bias2 = new double[outputSize];

        initWeights();
    }

    private void initWeights() {
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights1[i][j] = rand.nextGaussian();
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights2[i][j] = rand.nextGaussian();
            }
        }
    }

    private double[] sigmoid(double[] x) {
        double[] res = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            res[i] = 1 / (1 + Math.exp(-x[i]));
        }
        return res;
    }

    private double[] sigmoidDerivative(double[] x) {
        double[] res = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            res[i] = x[i] * (1 - x[i]);
        }
        return res;
    }

    private double[] feedForward(double[] input) {
        double[] hiddenLayerInput = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                hiddenLayerInput[i] += input[j] * weights1[j][i];
            }
            hiddenLayerInput[i] += bias1[i];
        }
        double[] hiddenLayerOutput = sigmoid(hiddenLayerInput);

        double[] outputLayerInput = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                outputLayerInput[i] += hiddenLayerOutput[j] * weights2[j][i];
            }
            outputLayerInput[i] += bias2[i];
        }
        return sigmoid(outputLayerInput);
    }

    private void backpropagation(double[] input, double[] target) {
        double[] hiddenLayerInput = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                hiddenLayerInput[i] += input[j] * weights1[j][i];
            }
            hiddenLayerInput[i] += bias1[i];
        }
        double[] hiddenLayerOutput = sigmoid(hiddenLayerInput);

        double[] outputLayerInput = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                outputLayerInput[i] += hiddenLayerOutput[j] * weights2[j][i];
            }
            outputLayerInput[i] += bias2[i];
        }
        double[] output = sigmoid(outputLayerInput);

        double[] outputError = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = target[i] - output[i];
        }

        double[] outputDelta = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputDelta[i] = outputError[i] * sigmoidDerivative(output)[i];
        }

        double[] hiddenError = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                hiddenError[i] += outputDelta[j] * weights2[i][j];
            }
        }

        double[] hiddenDelta = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            hiddenDelta[i] = hiddenError[i] * sigmoidDerivative(hiddenLayerOutput)[i];
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights2[i][j] += learningRate * outputDelta[j] * hiddenLayerOutput[i];
            }
        }
        for (int i = 0; i < outputSize; i++) {
            bias2[i] += learningRate * outputDelta[i];
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights1[i][j] += learningRate * hiddenDelta[j] * input[i];
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            bias1[i] += learningRate * hiddenDelta[i];
        }
    }

    public void train(double[][] inputs, double[][] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                backpropagation(inputs[i], targets[i]);
            }
            System.out.println("Epoch " + epoch + " completed");
        }
    }

    public double[] predict(double[] input) {
        return feedForward(input);
    }
}

