namespace bpnn_csharp;

/// <summary>
/// 突触
/// </summary>
/// <remarks>
/// 突触(Synapse)作为两个神经元之间的连接, 一般会有这几个构成部分:<br/>
/// 输入输出端: 连接前后两个神经元
/// 权重:一个初始随机的浮点值, 训练主要是练权重, 代表着每一个神经元在计算中的权重<br/>
/// </remarks>
public class Synapse
{
    public Synapse(Neuron input, Neuron outPut)
    {
        (Input, Output, Weight) = (input, outPut, NetWork.GetRandom());
        Input.Outputs.Add(this);
        Output.Inputs.Add(this);
    }
    /// <summary>
    /// 输入端
    /// </summary>
    public Neuron Input { get; set; }
    /// <summary>
    /// 输出端
    /// </summary>
    public Neuron Output { get; set; }
    /// <summary>
    /// 权重
    /// </summary>
    public double Weight { get; set; }
    /// <summary>
    /// 权重的变化量
    /// </summary>
    public double WeightDelta { get; set; }
}