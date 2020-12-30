package WordCount;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;


import org.apache.commons.collections.bag.SynchronizedSortedBag;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.util.EnumCounters;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.wml.dom.WMLHeadElementImpl;


public class NB_2018082017 {
    //region Java-API实现文件上传
//    public static void main(String[] args) throws IOException,InterruptedException,URISyntaxException {
//
//        //使用root权限，URI就是之前配置的路径:9000端口，配置相关信息
//        FileSystem fs = FileSystem.get(new URI("hdfs://192.168.29.144:9000"),new Configuration(),"root");
//        //指定源路径和目标路径
//        Path src = new Path("D://Hadoop实验专用测试文件夹//test.txt");
//        Path dst = new Path("/input_2018082017");
//        //上传文件到HDFS指定的目录下
//        fs.copyFromLocalFile(src,dst);
//        //关闭文件系统
//        fs.close();
//    }
    //endregion


    //MapReduc中的Driver类
//    public static void main(String[] args) throws IOException,ClassNotFoundException,InterruptedException{
//
//
//            //1.获取配置信息以及封装任务
//            Configuration configuration = new Configuration();
//            Job job = Job.getInstance(configuration);
//
//            //2.设置Jar加载路径
//            job.setJarByClass(NB_2018082017.class);
//
//            //3.设置map和reduce类
//            job.setMapperClass(WordCountMapper.class);
//            job.setReducerClass(WordCountReducer.class);
//
//            //4.设置map输出
//            job.setMapOutputKeyClass(Text.class);
//            job.setMapOutputValueClass(LongWritable.class);
//
//            //5.设置最终输出
//            job.setOutputKeyClass(Text.class);
//            job.setOutputValueClass(LongWritable.class);
//
//            //6.设置输入和输出路径
//            FileInputFormat.addInputPath(job,new Path(args[0]));
//            FileOutputFormat.setOutputPath(job,new Path(args[1]));
//
//            //7.提交
//            boolean result = job.waitForCompletion(true);
//            System.exit(result?0:1);
//        }
        //endregion
    public static void main(String[] args) throws Exception {
        predictALL();
    }

    private static String modelFilePath = "D:\\Hadoop实验专用测试文件夹\\2018082017_模型.txt";
    private static String testDataFilePath = "D:\\Hadoop实验专用测试文件夹\\test.txt";
    public static HashMap<String, Integer>parameters = null;     //情感标签集
    public static double Nd = 0;            //文件中的总记录数
    public static HashMap<String,Integer>allFeatures = null;    //整个训练样本的键值对
    public static HashMap<String,Double>labelFeatures = null;   //某一类别下所有词出现的总数
    public static HashSet<String> V = null;                      //总训练样本的不重复单词

    public static void loadModel(String modelFile) throws Exception {
        if (parameters!=null && allFeatures!=null){
            return;
        }
        parameters = new HashMap<String,Integer>(); //情感标签集
        allFeatures = new HashMap<String, Integer>();//全部属性对
        labelFeatures = new HashMap<String, Double>();//某一类别下所有词出现的总数
        V = new HashSet<String>();
        File file;
        BufferedReader br = new BufferedReader(new FileReader(modelFile));
        String line = null;


        /*已知模型文件的格式为：<情感标签String,出现次数Integer>，所以下文中选择HashMap结构存储结果*/
        while ((line = br.readLine())!=null){
            //获取模型文件中每行元素所包含的集合,包括类别计数（如：好评_一个 100）
            String feature = line.substring(0,line.indexOf("\t"));
            Integer count = Integer.parseInt(line.substring(line.indexOf("\t")+1));
            //过滤掉 类别计数 （好评：100000）和（差评：100000）
            if(feature.contains("_")){
                //allFeatures存储全部属性对
                allFeatures.put(feature,count);
                //获取各属性对中的类别
                String label = feature.substring(0,feature.indexOf("_"));
                //按照类别分别统计在各自类别下所有词出现的总数
                if(labelFeatures.containsKey(label)){
                    labelFeatures.put(label,labelFeatures.get(label)+count);
                }
                else {
                    labelFeatures.put(label,(double)count);
                }
                //拉普拉斯平滑处理，避免训练集中的零概率情况从而影响预测结果
                //如果在测试集中出现了在训练集里面从没有出现过的词语，那么就将它加入一个新的集合中
                String word = feature.substring(feature.indexOf("_")+1);
                if (!V.contains(word)){
                    V.add(word);
                }
            }
            else {
                parameters.put(feature,count);
                Nd += count;
            }
        }
        br.close();
    }


    public static String predict(ArrayList<String> sentence,String modelFile) throws Exception {
        loadModel(modelFile);
        String predLabel = null;
        double maxValue = Double.NEGATIVE_INFINITY; //最大类概率（默认值为负无穷小）
        //String[] words = sentence.split(" ");
        Set<String> labelSet = parameters.keySet(); //获得标签集
        for(String label:labelSet){
            double tempValue = Math.log(parameters.get(label)/Nd);  //先验概率
            /**
             * 先验概率P(c)= 类c下单词总数/整个训练样本的单词总数 parameters .get(label):类别c对应的文档在训练数据集中的计数
             * Nd:整个训练样本的单词总数
             */
            for(int i = 0;i<sentence.size();i++){
                String word = sentence.get(i);
                String lf = label + "_" + word;
                //计算最大似然概率
                if(allFeatures.containsKey(lf)){
                    //Laplace平滑，针对测试集中从未出现的情况，避免零概率影响预测结果，所以对
                    //最大似然概率公式进行修正，即分子加1，分母加上从未出现过词语的集合大小
                    //已知前提是各个特征条件相互独立，所以针对读取到的每行的似然概率我们可以这么求


                    //根据贝叶斯公式使用最大似然估计计算两种类别下的可能性
                    //某一类别（可能是好评可能是差评，取决于第一次读到的是什么类别）下的概率
                    tempValue += Math.log((double)(allFeatures.get(lf)+1)/(labelFeatures.get(label)+V.size()));
                }
                else {
                    //其对立类别下的概率
                    tempValue += Math.log((double)(1/(labelFeatures.get(label)+V.size())));
                }
            }
            //在文件的行数读取中，不断计算两个类别下的各个特征的似然概率，
            //最后比较得出较大的概率，那么此时的类别将作为
            //测试集的最后评判结果
            if(tempValue > maxValue){
                maxValue = tempValue;
                predLabel = label;
            }
        }
        return predLabel;
    }

    public static void predictALL() throws Exception {
        double accuracy = 0;        //计算正确的个数
        int amount = 0;             //计算测试结果集的数量
        try{
            File testResultFile = new File("D:\\Hadoop实验专用测试文件夹\\2018082017_测试结果.txt");
            File testDataFile  = new File(testDataFilePath);
            testResultFile.createNewFile();
            BufferedWriter writer = new BufferedWriter(new FileWriter(testResultFile));
            BufferedReader reader = new BufferedReader(new FileReader(testDataFile));
            String str1 = "";
            ArrayList<ArrayList<String>> testSet = new ArrayList<>();
            //读取测试集文件中的每行，加入到字符串数组中
            while ((str1 = reader.readLine())!=null){
                str1 = str1.replaceAll("\t"," ");
                StringTokenizer tokenizer = new StringTokenizer(str1);
                ArrayList<String> s = new ArrayList<>();
                while (tokenizer.hasMoreTokens()) {
                    s.add(tokenizer.nextToken());
                }
                testSet.add(s);
            }
//            for(int i = 0;i<testSet.size();i++){
//                System.out.println(testSet.get(i));
//            }
            ArrayList<String> res = new ArrayList<>();
            //对数组中的每一行数据进行预测，并将结果用数组存储
            for(int i = 0;i<testSet.size();i++)
            {
                ArrayList<String> s1 = testSet.get(i);
                String s2 = predict(s1,modelFilePath);
                amount++;
                res.add(s2);
            }
            //System.out.println(amount);
            //将预测结果（情感标签）写入文件
            for(int i = 0;i<res.size();i++)
            {
                writer.write(i+1+"\t"+res.get(i)+"\n");
                //System.out.println(i+1+"\t"+res.get(i));
            }
            //将预测结果与真实结果比对，最后给出预测的正确率
            for(int i = 0;i<testSet.size();i++){
                String value = testSet.get(i).get(0);
                if(res.get(i).equals(value)){
                    accuracy++;
                }
                System.out.println(i+1+" "+"真实情况:"+value+"\t预测结果:"+res.get(i));
            }
            //此处一定要把缓冲区中剩余数值全部输出，不然预测结果文件会不完整
            writer.flush();
            writer.close();
            //System.out.println(accuracy);
            double rate = (double)accuracy/testSet.size();
            System.out.println("预测正确率为："+rate*100+"%");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


