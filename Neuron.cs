using Collapsenav.Net.Tool;

namespace bpnn_csharp;
/// <summary>
/// 神经元
/// </summary>
/// <remarks>
/// 神经元(Neuron)是神经网络中最重要的组成部分之一, 一般会有这几个构成部分:<br/>
/// 输入神经元:上一层网络的神经元连接<br/>
/// 输出神经元:下一层网络的神经元连接<br/>
/// </remarks>
public class Neuron
{
    public Neuron(IActivationFunction actFun)
    {
        Inputs = new();
        Outputs = new();
        ActivationFunction = actFun;
        // 默认的随机偏置
        Bias = NetWork.GetRandom() / 10;
    }
    /// <summary>
    /// 值
    /// </summary>
    public double Value { get; set; }
    public IActivationFunction ActivationFunction { get; private set; }
    /// <summary>
    /// 偏置
    /// </summary>
    public double Bias { get; set; }
    public double BiasDelta { get; set; }
    /// <summary>
    /// 梯度
    /// </summary>
    public double Gradient { get; set; }
    public double GetValue()
    {
        if (Inputs.NotEmpty())
            Value = ActivationFunction.Forward(Inputs.Sum(i => i.Weight * i.Input.GetValue()) + Bias);
        return Value;
    }
    public double GetGradient(double? target = null)
    {
        if (!target.HasValue)
            Gradient = Outputs.Sum(i => i.Output.Gradient * i.Weight) * ActivationFunction.Back(Value);
        else
            Gradient = GetError(target.Value) * ActivationFunction.Back(Value);
        return Gradient;
    }
    public void Back(double target)
    {
        Gradient = GetError(target) * ActivationFunction.Back(Value);
    }
    public double GetError(double target)
    {
        return (target - Value) / 2;

        // var mean = (target + Value) / 2;
        // return ((mean - target) * (mean - target) + (mean - Value) * (mean - Value)) / 2;
        // return (target - Value) * (target - Value);
    }
    /// <summary>
    /// 更新权重
    /// </summary>
    /// <param name="lr"></param>
    public void UpdateWeight(double lr, double mont = 1)
    {
        var preDelta = BiasDelta;
        BiasDelta = lr * Gradient;
        Bias += BiasDelta;
        foreach (var synapse in Inputs)
        {
            preDelta = synapse.WeightDelta;
            synapse.WeightDelta = lr * Gradient * synapse.Input.Value;
            synapse.Weight += synapse.WeightDelta + preDelta * mont;
        }
    }
    public List<Synapse> Inputs { get; set; }
    public List<Synapse> Outputs { get; set; }
}