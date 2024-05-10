import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class KMeans {
    private final List<double[]> data;
    private final List<String> labels;
    private final int numAttributes;
    private final int k;
    private final List<double[]> centroids;
    private final List<List<Integer>> clusters;

    public KMeans(List<double[]> data, List<String> labels, int k) {
        this.data = data;
        this.labels = labels;
        this.numAttributes = data.get(0).length;
        this.k = k;
        this.centroids = initializeCentroids();
        this.clusters = new ArrayList<>();
    }
    private List<double[]> initializeCentroids() {
        List<double[]> centroids = new ArrayList<>();
        Random rand = new Random();
        Set<Integer> chosen = new HashSet<>();
        while (centroids.size() < k) {
            int idx = rand.nextInt(data.size());
            if (!chosen.contains(idx)) {
                centroids.add(data.get(idx));
                chosen.add(idx);
            }
        }
        return centroids;
    }
    private double calculateDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < numAttributes; i++) {
            sum += Math.pow(point1[i] - point2[i], 2);
        }
        return Math.sqrt(sum);
    }
    private void assignClusters() {
        clusters.clear();
        for (int i = 0; i < k; i++) {
            clusters.add(new ArrayList<>());
        }
        for (int i = 0; i < data.size(); i++) {
            double[] point = data.get(i);
            int clusterIdx = 0;
            double minDistance = Double.MAX_VALUE;
            for (int j = 0; j < k; j++) {
                double distance = calculateDistance(point, centroids.get(j));
                if (distance < minDistance) {
                    minDistance = distance;
                    clusterIdx = j;
                }
            }
            clusters.get(clusterIdx).add(i);
        }
    }
    private double updateCentroids() {
        double totalDistance = 0.0;
        for (int i = 0; i < k; i++) {
            List<Integer> cluster = clusters.get(i);
            for (int idx : cluster) {
                double[] point = data.get(idx);
                totalDistance += calculateDistance(point, centroids.get(i));
            }
        }
        for (int i = 0; i < k; i++) {
            double[] centroid = new double[numAttributes];
            List<Integer> cluster = clusters.get(i);
            for (int idx : cluster) {
                double[] point = data.get(idx);
                for (int j = 0; j < numAttributes; j++) {
                    centroid[j] += point[j];
                }
            }
            for (int j = 0; j < numAttributes; j++) {
                centroid[j] /= cluster.size();
            }
            centroids.set(i, centroid);
        }
        return totalDistance;
    }
    private void printClusterPurity() {
        for (int i = 0; i < k; i++) {
            Map<String, Integer> labelCounts = new HashMap<>();
            List<Integer> cluster = clusters.get(i);
            for (int idx : cluster) {
                String label = labels.get(idx);
                labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
            }
            StringBuilder purity = new StringBuilder("Cluster " + (i + 1) + ": ");
            for (Map.Entry<String, Integer> entry : labelCounts.entrySet()) {
                double percent = (double) entry.getValue() / cluster.size() * 100;
                purity.append(String.format("%.2f", percent)).append("% ").append(entry.getKey()).append(", ");
            }
            purity.setLength(purity.length() - 2);
            System.out.println(purity);
        }
    }
    public void run() {
        double prevDistance = Double.MAX_VALUE;
        while (true) {
            assignClusters();
            double totalDistance = updateCentroids();
            System.out.println("Sum of distances: " + totalDistance);
            printClusterPurity();
            if (Math.abs(prevDistance - totalDistance) < 0.0001) {
                break;
            }
            prevDistance = totalDistance;
        }
    }
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the path of the data file: ");
        String filePath = scanner.nextLine();

        List<double[]> data = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] attributes = new double[parts.length - 1];
                for (int i = 0; i < attributes.length; i++) {
                    attributes[i] = Double.parseDouble(parts[i]);
                }
                data.add(attributes);
                labels.add(parts[parts.length - 1]);
            }
        } catch (IOException e) {
            System.out.println("Error!!");
            return;
        }
        System.out.print("Enter the number of clusters (k): ");
        int k = scanner.nextInt();
        KMeans kMeans = new KMeans(data, labels, k);
        kMeans.run();
    }
}