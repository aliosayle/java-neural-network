import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import javax.swing.*;

public class Main {
    public static void main(String[] args) throws IOException {
        String trainImagesPath = "archive/train-images.idx3-ubyte";
        String trainLabelsPath = "archive/train-labels.idx1-ubyte";
        String testImagesPath = "archive/t10k-images.idx3-ubyte";
        String testLabelsPath = "archive/t10k-labels.idx1-ubyte";

        // Load training data
        double[][] trainImages = loadMNISTImages(trainImagesPath);
        double[][] trainLabels = loadMNISTLabels(trainLabelsPath);

        // Define the architecture: 784 inputs (28x28), 64 hidden neurons, 10 outputs (digits 0-9)
        NeuralNetwork nn = new NeuralNetwork(784, 64, 10);

        // Train the network on the actual MNIST training data
        nn.train(trainImages, trainLabels, 10);  // Train for 10 epochs

        // Predict using a specific training image (for example, image at index 0)
        int imageIndex = 0;  // Change this to test other images
        double[] prediction = nn.predict(trainImages[imageIndex]);

        // Get the actual label for the image
        int actualLabel = getActualLabel(trainLabels[imageIndex]);

        // Output the prediction result (probabilities for each digit)
        System.out.println("Predicted output: ");
        for (int i = 0; i < prediction.length; i++) {
            System.out.printf("Digit %d: %.3f%n", i, prediction[i]);
        }

        // Determine the predicted digit
        int predictedDigit = getPredictedDigit(prediction);
        System.out.println("Predicted digit: " + predictedDigit);
        System.out.println("Actual digit: " + actualLabel);

        // Check if the prediction was correct
        if (predictedDigit == actualLabel) {
            System.out.println("Prediction is correct!");
        } else {
            System.out.println("Prediction is incorrect.");
        }

        // Optional: Display the image for visual confirmation
        displayImageFromMNIST(trainImagesPath, imageIndex);
    }

    // Helper function to determine which digit has the highest predicted value
    private static int getPredictedDigit(double[] prediction) {
        int predictedDigit = 0;
        double maxProbability = prediction[0];
        for (int i = 1; i < prediction.length; i++) {
            if (prediction[i] > maxProbability) {
                maxProbability = prediction[i];
                predictedDigit = i;
            }
        }
        return predictedDigit;
    }

    // Helper function to get the actual label from the one-hot encoded array
    private static int getActualLabel(double[] label) {
        for (int i = 0; i < label.length; i++) {
            if (label[i] == 1.0) {
                return i;  // The index of the label corresponds to the actual digit
            }
        }
        return -1;  // Should not happen if labels are correctly formatted
    }

    // Load MNIST images from the IDX file
    private static double[][] loadMNISTImages(String path) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(path));

        // Read the header info (magic number, number of images, rows, columns)
        int magicNumber = dis.readInt();
        int numberOfImages = dis.readInt();
        int rows = dis.readInt();
        int cols = dis.readInt();

        int imageSize = rows * cols;  // 28 * 28 = 784 pixels per image
        double[][] images = new double[numberOfImages][imageSize];

        // Read each image and normalize pixel values (0-255) to (0-1)
        for (int i = 0; i < numberOfImages; i++) {
            for (int j = 0; j < imageSize; j++) {
                images[i][j] = dis.readUnsignedByte() / 255.0;  // Normalize to [0, 1]
            }
        }

        dis.close();
        return images;
    }

    // Load MNIST labels from the IDX file
    private static double[][] loadMNISTLabels(String path) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(path));

        // Read the header info (magic number, number of labels)
        int magicNumber = dis.readInt();
        int numberOfLabels = dis.readInt();

        // Labels are one-hot encoded (0-9 to a vector of size 10)
        double[][] labels = new double[numberOfLabels][10];

        for (int i = 0; i < numberOfLabels; i++) {
            int label = dis.readUnsignedByte();  // Read the label (0-9)
            labels[i][label] = 1.0;  // One-hot encode the label
        }

        dis.close();
        return labels;
    }

    // Display an image from the MNIST dataset
    private static void displayImageFromMNIST(String path, int index) throws IOException {
        double[][] images = loadMNISTImages(path);
        double[] image = images[index];

        BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int pixel = (int) (image[i * 28 + j] * 255); // Scale back to [0, 255]
                int rgb = pixel << 16 | pixel << 8 | pixel; // Convert to RGB
                bufferedImage.setRGB(j, i, rgb);
            }
        }

        // Create a JFrame to display the image
        JFrame frame = new JFrame("MNIST Image");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 300);
        JLabel label = new JLabel(new ImageIcon(bufferedImage));
        frame.add(label);
        frame.setVisible(true);
    }
}
