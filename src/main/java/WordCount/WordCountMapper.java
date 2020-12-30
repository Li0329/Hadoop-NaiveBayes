package WordCount;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.regex.Pattern;

/**
 * @Author:The King Of the World
 * @Date:2020/12/26
 * @Description:WordCount
 * @version：1.0
 */
public class WordCountMapper extends Mapper<LongWritable, Text,Text, LongWritable> {
    //region 生成模型文件
    private static LongWritable one = new LongWritable(1);

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

        //获取每行文档内容，并且进行拆分，分成类别+特征组的形式
        String content[] = value.toString().split("\t",2);
        String label = content[0];
        //获取当前特征集合
        String[] features = content[1].split(" ");

        //输出类别下特征计数
        for(String feature:features){
            //过滤掉包含非中文字符和全部都是非中文字符构成的词语
            if (Pattern.matches("[\u4E00-\u9FA5]+",feature))
            {
                context.write(new Text(label+" "+feature),one);
            }
        }

        //输出类别计数
        context.write(new Text(label),one);
    }
    //endregion

}
