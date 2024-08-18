using Collapsenav.Net.Tool;

namespace bpnn_csharp;

/// <summary>
/// 神经网络
/// </summary>
public class NetWork
{
    public static Random Rand = new();
    public NetWork()
    {
        Layers = new();
    }
    public List<Layer> Layers { get; set; }
    /// <summary>
    /// 输入层
    /// </summary>
    public Layer Input => Layers.First();
    /// <summary>
    /// 输出层
    /// </summary>
    public Layer Output => Layers.Last();
    /// <summary>
    /// 添加一层网络
    /// </summary>
    /// <param name="size">神经元数量</param>
    /// <param name="actFun">激活函数</param>
    /// <returns></returns>
    public NetWork AddLayer(int size, IActivationFunction actFun)
    {
        // 如果添加了第一层, 直接进行初始化即可
        if (Layers.IsEmpty())
            Layers.Add(new Layer(size, actFun));
        // 如果是之后的层, 则需要将前后两次进行连接
        else
        {
            var previous = Output;
            Layers.Add(new Layer(size, actFun));
            // 连接前后两次神经元
            Output.Previous = previous;
            previous.Next = Output;
        }
        return this;
    }
    public IEnumerable<double> Forward(double[] inputs)
    {
        Input.SelectWithIndex().ForEach(i => i.value.Value = inputs[i.index]);
        return Output.Select(i => i.GetValue()).ToList();
    }
    public void Back(double[] target, double lr, double mont = 1)
    {
        // 计算所有神经元的梯度, GetGradient 内部有递归处理
        Output.SelectWithIndex().ForEach(kv => kv.value.GetGradient(target[kv.index]));
        // 先将网络反过来
        Layers.Reverse();
        // 然后从后往前一层一层更新偏置和权重
        Layers.ForEach(i => i.ForEach(ii => ii.UpdateWeight(lr, mont)));
        // 更新完成之后再度翻转, 使得网络恢复原状
        Layers.Reverse();
    }
    public double GetError(double[] target) => Output.SelectWithIndex().Sum(i => i.value.GetError(target[i.index]));
    /// <summary>
    /// 使用统一的随机数生成, 便于通过种子固定
    /// </summary>
    /// <returns></returns>
    public static double GetRandom() => Rand.NextDouble() / 10;
    /// <summary>
    /// 设置随机数种子
    /// </summary>
    /// <param name="seed"></param>
    public void SetRandSeed(int seed) => Rand = new Random(seed);
}
