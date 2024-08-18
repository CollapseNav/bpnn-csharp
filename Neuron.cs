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
    /// <summary>
    /// 简单初始化神经元
    /// </summary>
    public Neuron(IActivationFunction actFun)
        => (Inputs, Outputs, ActivationFunction, Bias) = (new List<Synapse>(), new List<Synapse>(), actFun, NetWork.GetRandom());
    /// <summary>
    /// 值
    /// </summary>
    public double Value { get; set; }
    /// <summary>
    /// 激活函数
    /// </summary>
    public IActivationFunction ActivationFunction { get; private set; }
    /// <summary>
    /// 偏置
    /// </summary>
    public double Bias { get; set; }
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
        // 最后一层直接计算
        if (target.HasValue)
            Gradient = GetError(target.Value) * ActivationFunction.Back(Value);
        // 如果存在输出神经元, 该神经元的梯度为后一层的梯度加权求和乘以偏导
        if (Outputs.NotEmpty())
            Gradient = Outputs.Sum(i => i.Output.Gradient * i.Weight) * ActivationFunction.Back(Value);
        // 如果存在输入神经元, 则计算前一层的梯度
        if (Inputs.NotEmpty())
            Inputs.ForEach(i => i.Input.GetGradient());
        return Gradient;
    }
    public double GetError(double target) => target - Value;
    /// <summary>
    /// 更新权重
    /// </summary>
    /// <param name="lr"></param>
    public void UpdateWeight(double lr, double mont = 1)
    {
        // 计算权重变化量
        var BiasDelta = lr * Gradient;
        // 调整神经元的偏置
        Bias += BiasDelta;
        foreach (var synapse in Inputs)
        {
            // 更新突触权重
            var preDelta = synapse.WeightDelta;
            synapse.WeightDelta = BiasDelta * synapse.Input.Value;
            // 使用上一次的变化量和动量加快收敛
            synapse.Weight += synapse.WeightDelta + preDelta * mont;
        }
    }
    /// <summary>
    /// 输入
    /// </summary>
    public List<Synapse> Inputs { get; set; }
    /// <summary>
    /// 输出
    /// </summary>
    public List<Synapse> Outputs { get; set; }
}
