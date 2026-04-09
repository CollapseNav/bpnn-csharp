/// <summary>
/// 归一化
/// </summary>
public class Normalizer
{
    /// <summary>
    /// 输入数据的最小值
    /// </summary>
    public double[]? InputMin { get; private set; }
    /// <summary>
    /// 输入数据的最大值
    /// </summary>
    public double[]? InputMax { get; private set; }
    /// <summary>
    /// 输出数据的最小值
    /// </summary>
    public double[]? OutputMin { get; private set; }
    /// <summary>
    /// 输出数据的最大值
    /// </summary>
    public double[]? OutputMax { get; private set; }
    /// <summary>
    /// 计算输入输出的最大最小值
    /// </summary>
    /// <param name="data"></param>
    public void Fit(IData[] data)
    {
        if (data.Length == 0) return;
        InputMin = new double[data[0].GetInput().Length];
        InputMax = new double[data[0].GetInput().Length];
        OutputMin = new double[data[0].GetOutput().Length];
        OutputMax = new double[data[0].GetOutput().Length];
        // 计算输入数据的最大最小值
        for (int i = 0; i < data[0].GetInput().Length; i++)
        {
            InputMin[i] = data[0].GetInput()[i];
            InputMax[i] = data[0].GetInput()[i];
            foreach (var row in data)
            {
                if (row.GetInput()[i] < InputMin[i]) InputMin[i] = row.GetInput()[i];
                if (row.GetInput()[i] > InputMax[i]) InputMax[i] = row.GetInput()[i];
            }
        }
        // 计算输出数据的最大最小值
        for (int i = 0; i < data[0].GetOutput().Length; i++)
        {
            OutputMin[i] = data[0].GetOutput()[i];
            OutputMax[i] = data[0].GetOutput()[i];
            foreach (var row in data)
            {
                if (row.GetOutput()[i] < OutputMin[i]) OutputMin[i] = row.GetOutput()[i];
                if (row.GetOutput()[i] > OutputMax[i]) OutputMax[i] = row.GetOutput()[i];
            }
        }
    }

    /// <summary>
    /// 将数据进行归一化操作
    /// </summary>
    /// <param name="sample"></param>
    /// <returns></returns>
    public double[] Transform(IData sample)
    {
        var origin = sample.GetInput();
        double[] normalized = new double[origin.Length];
        for (int i = 0; i < origin.Length; i++)
        {
            double range = InputMax[i] - InputMin[i];
            if (range == 0)
                normalized[i] = 0.5; // 如果该特征无变化，设为中间值
            else
                normalized[i] = (origin[i] - InputMin[i]) / range;
            origin[i] = normalized[i]; // 记录归一化后的值
        }
        origin = sample.GetOutput();
        normalized = new double[origin.Length];
        for (int i = 0; i < origin.Length; i++)
        {
            double range = OutputMax[i] - OutputMin[i];
            if (range == 0)
                normalized[i] = 0.5; // 如果该特征无变化，设为中间值
            else
                normalized[i] = (origin[i] - OutputMin[i]) / range;
            origin[i] = normalized[i]; // 记录归一化后的值
        }

        return normalized;
    }
    /// <summary>
    /// 反归一化，还原数据
    /// </summary>
    /// <param name="normalizedValue"></param>
    /// <param name="featureIndex"></param>
    /// <returns></returns>
    public double InverseTransform(double normalizedValue, int featureIndex)
    {
        double range = OutputMax[featureIndex] - OutputMin[featureIndex];
        if (range == 0)
            return OutputMin[featureIndex];
        return normalizedValue * range + OutputMin[featureIndex];
    }
}