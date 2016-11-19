/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekatubes;


import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


/**
 *
 * @author Kris
 */
public class Wekatubes {

    public static void reEvaluateModel(Instances test, Classifier modelSet, Instances trainSet) throws Exception{
        Evaluation eval = new Evaluation(trainSet);
        StringBuffer predsBuffer = new StringBuffer();
        PlainText plainText = new PlainText();
        plainText.setHeader(trainSet);
        plainText.setBuffer(predsBuffer);
        
        eval.evaluateModel(modelSet,test, plainText);
        System.out.println("\nRESULT\n");
        System.out.println("=== Predictions on user test set ===\n" +"\n" +"    inst#     actual  predicted error prediction");
        System.out.println(predsBuffer.toString());
        System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString()); 
    }
    
    public static void naiveBayes(String path ) throws Exception {
        DataSource source = new DataSource(path);
	Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);
        naiveBayes nb = new naiveBayes();
        nb.buildClassifier(dataset);
        System.out.println("=== Naive Bayes ===");
        for (int i = 0 ; i < nb.classCount; i++) {
            System.out.print(dataset.classAttribute().value(i)+ "\t\t\t");
        }
        System.out.println("Result");
        Evaluation ev = new Evaluation(dataset);
        ev.evaluateModel(nb, dataset);
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
        return testSet;
    }
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
    }
    
}
