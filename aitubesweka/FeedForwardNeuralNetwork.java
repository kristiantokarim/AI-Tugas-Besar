package aitubesweka;


import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class FeedForwardNeuralNetwork extends AbstractClassifier {
	private Instances proInstances;
    private int classCount;
    private int attrCount;
    private NeuralNetwork nn;
	
	public void buildClassifier(Instances ins) throws Exception {
		// TODO Auto-generated method stub
		if (ins.checkForStringAttributes()) {
            throw new Exception("Can't handle string attributes!");
        }
        if (ins.classAttribute().isNumeric()) {
            throw new Exception("Naive Bayes: Class is numeric!");
        }
      
        proInstances = new Instances(ins,0);
        classCount = ins.numClasses();
        attrCount = ins.numAttributes() - 1;
        
        
        
        
        
        
        System.out.println(ins);
        //System.out.println(proInstances);
        //System.out.println(classCount);
        //System.out.println(attrCount);
        Attribute proAttr = ins.attribute(1);
        
        System.out.println("=====");
        Instance proIns;
        double[] instance;
        double[] check;
        double[] input = new double[attrCount];
        double[] target = new double[classCount];
        nn = new NeuralNetwork(attrCount,2,classCount);
        for (int i =0; i < ins.numInstances(); i++) {
        	proIns = ins.get(i);
        	instance = proIns.toDoubleArray();
        	for (int j = 0; j < instance.length - 1; j++) {
        		input[j] = instance[j];
        	}
        	for (int j = 0; j < classCount; j++) {
        		if(((int) instance[attrCount]) != j) {
        			target[j] = 0;
        		}
        		else {
        			target[j] = 1;
        		}
        		System.out.println(i + ": target["+j+"] = "+target[j]);
        	}
        	
        	//Initialize Input
        	nn.setInput(input);
            nn.setTarget(target);
            //Feed Forward
            nn.countOutput();
            //Back Propagation
            nn.countOutputLayerError();
            nn.setLearningRate(0.1);
            nn.updateOutputLayerWeight();
            nn.countFirstLayerError();
            nn.updateFirstLayerWeight();
            nn.printAllWeight();
        	
        }
	}
	
	public double[] classifyingInstance(Instance instance) throws Exception {
		double[] result;
		double[] arrayInstance;
		arrayInstance = instance.toDoubleArray();
		double[] input = new double[instance.numAttributes()-1];
		for (int j = 0; j < arrayInstance.length - 1; j++) {
    		input[j] = arrayInstance[j];
    	}
		nn.setInput(input);
		result = nn.countOutput();
		for(int i = 0; i < result.length; i++) {
			System.out.println("result = " +result[i]);
		}
		return result;
	}
	
	public double[] distributionForInstance(Instance instance) throws Exception {
		return classifyingInstance(instance);
	  }
	
}

