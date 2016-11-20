/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekatubes;


import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 *
 * @author Kris
 */
public class Wekatubes {
    
    public static void main(String[] args) throws Exception {
        Scanner in = new Scanner(System.in);
        System.out.print("Masukan lokasi data : ");
        String path = in.next();
        DataSource source = new DataSource(path);
        Instances train = source.getDataSet();
        Instances read = source.getDataSet();
        System.out.print("Masukan nama kelas : ");
        String className = in.next();
        train.setClass(train.attribute(className));
        Discretize dis = new Discretize();
        dis.setInputFormat(train);
        train = Filter.useFilter(train, dis);
        System.out.println("1. Buat model klasifikasi");
        System.out.println("2. Evaluasi test data dengan model yang sudah ada");
        int pil = in.nextInt();
        Classifier cls = null;
        switch (pil) {
            case 1 :
                cls = new naiveBayes();
                cls.buildClassifier(train);
                System.out.print("Simpan model ?");
                String input = in.next();
                if ("y".equals(input)) {
                    System.out.print("Masukan lokasi penyimpanan : ");
                    path = in.next();
                    path += "\\" + (new SimpleDateFormat("dd-MM-yyyy_HHmmss").format(Calendar.getInstance().getTime()))+".model";
                    weka.core.SerializationHelper.write(path, cls);
                    System.out.println("Model telah tersimpan");
                }
                break;
            case 2 :
                System.out.print("Masukan lokasi model : ");
                path = in.next();
                cls = (Classifier) weka.core.SerializationHelper.read(path);
                break;
        }
        System.out.println("Evaluasi dengan : ");
        System.out.println("1. 10-folds Cross Validation");
        System.out.println("2. Evaluate Model");
        pil = in.nextInt();
        System.out.println("Test data : ");
        System.out.println("1. Split test");
        System.out.println("2. Full training set");
        System.out.println("3. Input test data from file");
        System.out.println("4. Input test data from System.in");
        int testData = in.nextInt();
        Instances test = new Instances(train,0);
        switch (testData) {
            case 1 :
                System.out.print("Masukan persentasi : ");
                double percent = in.nextDouble();
                int trainSize = (int) Math.round(train.numInstances() * percent / 100);
                int testSize = train.numInstances() - trainSize;
                test = new Instances(train, trainSize, testSize);
                train = new Instances(train, 0, trainSize);
                break;
            case 2 :
                test = new Instances(read,0, train.numInstances());
                break;
            case 3 :
                System.out.print("Masukan lokasi data : ");
                path = in.next();
                DataSource testSource = new DataSource(path);
                test = testSource.getDataSet();
                break;
            case 4 :
                System.out.print("Masukan jumlah data yang ingin diinput : ");
                int numInst = in.nextInt();
                test = new Instances(read);
                test.clear();
                int numAtt = test.numAttributes();
                Instance tempData = new DenseInstance(numAtt);
                tempData.setDataset(test);
                for (int i = 0 ; i < numInst ; i++) {
                    System.out.println("Masukan nilai data " + (i+1) +" : ");
                    for (int j = 0; j < numAtt ; j++) {
                        System.out.print("\t" + (j+1) + ". " + train.attribute(j).name() + " : ");
                        if ("numeric".equals(Attribute.typeToString(read.attribute(j).type()))) {
                            double attVal = in.nextDouble();
                            tempData.setValue(j, attVal);
                        }
                        else {
                            String attVal = in.next();
                            tempData.setValue(j,attVal);
                        }
                    }
                    test.add(tempData);
                }
                break;
        }
        test = Filter.useFilter(test, dis);
        Evaluation eval = new Evaluation(train);
        switch (pil) {
            case 1 :
                eval.crossValidateModel(cls, test,10, new Random(1));
                break;
            case 2 :
                NaiveBayes n = new NaiveBayes();
                n.buildClassifier(train);
                eval.evaluateModel(n, test);
                break;
        }
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
        
    }
    
}
