using System.Collections;
using Collapsenav.Net.Tool;

namespace bpnn_csharp;
/// <summary>
/// 神经元层
/// </summary>
public class Layer : IEnumerable<Neuron>
{
    public IActivationFunction ActivationFunction { get; private set; }
    public Layer(int size, IActivationFunction actFunc)
    {
        Neurons = new Neuron[size];
        ActivationFunction = actFunc;
        foreach (var i in Enumerable.Range(0, size))
            Neurons[i] = new Neuron(ActivationFunction);
    }
    /// <summary>
    /// 上一层
    /// </summary>
    public Layer? Previous
    {
        get => previous; set
        {
            previous = value;
            // 自动创建突触连接两次神经元
            if (previous != null)
                Neurons.ForEach(n => previous.Neurons.ForEach(p => _ = new Synapse(p, n)));
        }
    }
    private Layer? previous;
    /// <summary>
    /// 下一层
    /// </summary>
    public Layer? Next { get; set; }
    /// <summary>
    /// 本层的神经元
    /// </summary>
    public Neuron[] Neurons { get; private set; }
    public IEnumerator<Neuron> GetEnumerator()
    {
        return ((IEnumerable<Neuron>)Neurons).GetEnumerator();
    }
    IEnumerator IEnumerable.GetEnumerator()
    {
        return Neurons.GetEnumerator();
    }
}
