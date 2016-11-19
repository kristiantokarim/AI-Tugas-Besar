/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aitubesweka;

import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SerializationHelper;

/**
 *
 * @author Kris
 */
public class mainNB {

     public static void reEvaluateModel(Instances test, Classifier modelSet, Instances trainSet) throws Exception{
        Evaluation eval = new Evaluation(trainSet);
        StringBuffer predsBuffer = new StringBuffer();
        PlainText plainText = new PlainText();
        plainText.setHeader(trainSet);
        plainText.setBuffer(predsBuffer);
        
        
        Discretize dis = new Discretize();
        System.out.println(filteredDataset.toString()+"\n");
        System.out.println("pass 1\n");
        
        dis.setInputFormat(filteredDataset);
        
        
        System.out.println(test.toString()+"\n");
        System.out.println("pass 2\n");
        
        test = Filter.useFilter(test, dis);
        
        System.out.println(test.toString()+"\n");
        System.out.println("pass 3\n");
        
        eval.evaluateModel(modelSet,test, plainText);
        
        System.out.println("\nRESULT\n");
        System.out.println("=== Predictions on user test set ===\n" +"\n" +"    inst#     actual  predicted error prediction");
        System.out.println(predsBuffer.toString());
        System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString()); 
    }
	
    static naiveBayes nb;
    static Instances dataset, filteredDataset; 
    static Classifier a;
    
    
    public static void naiveBayes(String path ) throws Exception {
        DataSource source = new DataSource(path);
	dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);
        Discretize dis = new Discretize();
        dis.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, dis);
        
        System.out.println("====================================");
        System.out.println(dataset.toString()+"\n");
        filteredDataset = dataset;
        nb = new naiveBayes();
        nb.buildClassifier(dataset);
        
        System.out.println("=== Naive Bayes ===");
        for (int i = 0 ; i < nb.classCount; i++) {
            System.out.print(dataset.classAttribute().value(i)+ "\t\t");
        }
        System.out.println("Result");
        Evaluation ev = new Evaluation(dataset);
        ev.crossValidateModel(nb, dataset,10,new Random(1));
        System.out.println(nb.resultString(ev));
    }
    public static void save(String path, Classifier cls) throws Exception {
        weka.core.SerializationHelper.write(path, cls);
    }
    
    public static Classifier read(String path) throws Exception {
        return (Classifier) weka.core.SerializationHelper.read(path);
    }
    public static Instances readInstance(Instances ins, Scanner in){
        int numAtt =  ins.numAttributes();
        Instances testSet = new Instances(ins);
        Instance newData = new DenseInstance(numAtt);
        String b, ans = "Y";
        double a;
        boolean cont = true;
        testSet.clear();
        newData.setDataset(testSet);
        while (ans.equals("Y")){
            System.out.println("\nMasukkan nilai atribut :");
            for (int i=0; i<numAtt;i++){
                String type = (Attribute.typeToString(ins.attribute(i)));
                System.out.print(" >"+(i+1)+". "+ins.attribute(i).name()+": ");
                if ("numeric".equals(type)) {
                    a = in.nextDouble();
                    newData.setValue(i, a);
                }
                else{
                    b = in.next();
                    newData.setValue(i, b);
                }
            }
        testSet.add(newData);
        System.out.print("\nApakah Anda ingin memasukkan data lain (Y/N)?: ");
        ans = in.next();
        }
        System.out.println(testSet.toString()+"\n-------------------------------\n");
        return testSet;
    }
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        naiveBayes("F:\\iris.arff");
        save("F:\\iris.model",nb);
        
        DataSource source = new DataSource("F:\\iris.arff");
        System.out.println("pass 1");
	dataset = source.getDataSet();
        System.out.println("pass 1a");
        dataset.setClassIndex(dataset.numAttributes()-1);
        System.out.println("pass 1v");
        
       
        
        System.out.println("pass 1");
        
        nb = (aitubesweka.naiveBayes) read("F:\\iris.model");
        
        Scanner in = new Scanner(System.in);
        System.out.println("pass 1");
        reEvaluateModel(readInstance(dataset,in), nb, dataset);
        System.out.println("pass 1");
    }
    
    
}
