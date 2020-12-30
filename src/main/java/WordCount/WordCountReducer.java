package WordCount;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * @Author:The King Of the World
 * @Date:2020/12/26
 * @Description:WordCount
 * @version：1.0
 */
public class WordCountReducer extends Reducer<Text, LongWritable,Text,LongWritable> {


    Long sum;
    LongWritable v = new LongWritable();
    @Override
    protected void reduce(Text key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
        //累加求和
        sum = 0L;
        for(LongWritable count:values){
            sum += count.get();
        }

        //输出
        v.set(sum);
        context.write(key,v);
    }
}
