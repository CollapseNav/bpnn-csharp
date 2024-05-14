using Collapsenav.Net.Tool;

namespace bpnn_csharp;

/// <summary>
/// 神经网络
/// </summary>
class NetWork
{
    public static Random Rand = new(0);
    public NetWork()
    {
        Layers = new();
    }
    public List<Layer> Layers { get; set; }
    public Layer Input => Layers.First();
    public Layer Output => Layers.Last();
    /// <summary>
    /// 添加一层网络
    /// </summary>
    /// <param name="size">神经元数量</param>
    /// <param name="actFun">激活函数</param>
    /// <returns></returns>
    public NetWork AddLayer(int size, IActivationFunction actFun)
    {
        if (Layers.IsEmpty())
            Layers.Add(new Layer(size, actFun));
        else
        {
            var previous = Output;
            Layers.Add(new Layer(size, actFun));
            Output.Previous = previous;
            previous.Next = Output;
        }
        return this;
    }
    public void Forward(double[] inputs)
    {
        Input.SelectWithIndex().ForEach(i => i.value.Value = inputs[i.index]);
        Output.ForEach(i => _ = i.GetValue());
    }
    public void Back(double[] target, double lr, double mont = 1)
    {
        Output.SelectWithIndex().ForEach(kv => kv.value.GetGradient(target[kv.index]));
        Layers.Reverse();
        Layers.Skip(1).ForEach(i => i.ForEach(ii => ii.GetGradient()));
        Layers.ForEach(i => i.ForEach(ii => ii.UpdateWeight(lr, mont)));
        Layers.Reverse();
    }
    public double GetError(double target)
    {
        return Output.Sum(i => i.GetError(target));
    }
    /// <summary>
    /// 使用统一的随机数生成, 便于通过种子固定
    /// </summary>
    /// <returns></returns>
    public static double GetRandom() => Rand.NextDouble() / 10;
    public static void SetRandSeed(int seed) => Rand = new Random(seed);
}
