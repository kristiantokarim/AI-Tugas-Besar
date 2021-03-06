package aitubesweka;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Test {
	
	
	public static Instances filterNumericToNominal(Instances dataset, String[] opts) throws Exception {
		NumericToNominal numToNom = new NumericToNominal();
		numToNom.setOptions(opts);
		numToNom.setInputFormat(dataset);
		Instances nomData = Filter.useFilter(dataset, numToNom);
		return nomData;
	}
	
	public static Instances filterDiscretize(Instances dataset, String[] opts) throws Exception {
		Discretize discretize = new Discretize();
		discretize.setOptions(opts);
		discretize.setInputFormat(dataset);
		Instances numData = Filter.useFilter(dataset, discretize);
		return numData;
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		double[] tes = new double[0];
		// TODO Auto-generated method stub
		//Read Data Set and assign to dataset
		DataSource source = new DataSource("/home/cmrudi/weka-3-8-0/data/iris.arff");
		Instances dataset = source.getDataSet();
		//String[] discOpts = new String[]{"-B","5","-R","5"};
		//dataset = filterDiscretize(dataset, discOpts);	
		//System.out.println(dataset);
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		FeedForwardNeuralNetwork FFNN = new FeedForwardNeuralNetwork();
		FFNN.buildClassifier(dataset);
		
		Instance ins;
		for (int i = 1; i < 150; i++) {
			System.out.println(i+":");
			FFNN.classifyingInstance(dataset.get(i));
		}
		weka.core.SerializationHelper.write("saveFFNN.model", FFNN);
		FeedForwardNeuralNetwork loadFFNN = (FeedForwardNeuralNetwork) weka.core.SerializationHelper.read("saveFFNN.model");
		
		//loadFFNN.classifyingInstance(ins);
		
		//Evaluation ev = new Evaluation(dataset);
		//ev.evaluateModel(FFNN,dataset);
		//System.out.println(ev.toSummaryString());
		
	}

}

