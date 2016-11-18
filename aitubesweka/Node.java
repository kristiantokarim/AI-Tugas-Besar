package aitubesweka;

public class Node {
	
	private static final int MaxNumberOfInput = 100;
	private static final double InitialWeight = 1;
	private double learningRate;
	private double[] weight;
	private double currentOutput;
	private double currentError;
	private double currentTarget; //Applicable only for output node
	
	/*
	 * Input : Jumlah input menuju node tersebut
	 * Proses : Inisialisasi weight setiap edge menuju node tersebut dengan InitialWeight 
	 * 
	 */
	public Node(int numberOfInput) {
		weight = new double[numberOfInput];
		for (int i = 0; i < numberOfInput; i++) {
			weight[i]= InitialWeight;
		}
	}
	
	/*
	 * Input : Array of Double yang berisi input menuju Node
	 * Proses : Menghitung output yang diberikan node terseut menggunakan array input dan array weight
	 *
	 */
	
	public double countOutput(double[] input) {
		double sum = 0;
		for (int i =0; i < input.length; i++) {
			sum = sum + input[i] * weight[i];
		}
		currentOutput = 1/(1+Math.pow(Math.E, (-1)*sum));
		return currentOutput ;
	}
	
	/*
	 * 
	 * Input : target dari output Node
	 * Proses : mengembalikan error pada node yang menggunakan fungsi sigmoid
	 * Note : hanya berlaku untuk output layer
	 * 
	 */
	
	public double  countErrorOutputLayer() {
		currentError = currentOutput*(1-currentOutput)*(currentTarget - currentOutput);
		return currentError;
	}
	
	public double countErrorFirstLayer(double weightxerror) {
		currentError = currentOutput*(1-currentOutput)*weightxerror;
		return currentError;
	}
	
	/*
	 * 
	 * Input : Index yang ingin diubah weightnya dan inputValue adalah weight baru
	 * Proses : Mengganti weight lama dengan weight baru
	 * 
	 */
	
	public void setWeight(int inputIndex, double inputValue) {
		weight[inputIndex] = inputValue;
	}
	
	/*
	 * 
	 * Input : Index array weight, dimulai dari nol
	 * 
	 */
	
	public double getWeight(int inputIndex) {
		return weight[inputIndex];
	}
	
	/*
	 * 
	 * Input : nilai target
	 * Output : memberikan nilai pada attribut target
	 * Note : hanya berlaku untuk output node
	 * 
	 */
	
	public void setTarget(double targetValue) {
		currentTarget = targetValue;
	}
	
	/*
	 * 
	 * Getter untuk target
	 * 
	 */
	
	public double getTarget() {
		return currentTarget;
	}
	
	/*
	 * 
	 * Setter untuk learning rate
	 * 
	 */
	
	public void setLearningRate(double value) {
		learningRate = value;
	}
	
	/*
	 * 
	 * Mengupdate seluruh weight yang menuju node tersebut
	 * 
	 */
	
	public void updateWeight(double[] input) {
		if (input.length == weight.length) {
			for (int i = 0; i < weight.length; i++) {
				weight[i] = weight[i] + learningRate*currentError*input[i];
			}
		}else  {
			System.out.println("On Node class, method updateWeight, weight length not match with input lenght");
		}
	}
	
	/*
	 * 
	 * getter untuk currentError
	 * 
	 */
	
	public double getError() {
		return currentError;
	}
	
	/*
	 * 
	 * getter untuk currentOutput
	 * 
	 */
	
	public double getOutput() {
		return currentOutput;
	}
	
	/*
	 * 
	 * Mencetak weight yang menuju node
	 * 
	 */
	
	public void printWeight() {
		for(int i = 0; i < weight.length; i++) {
			System.out.print(weight[i] + "|");
		}
		System.out.println();
	}
}
