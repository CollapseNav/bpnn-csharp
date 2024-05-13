namespace bpnn_csharp;

/// <summary>
/// 激活函数
/// </summary>
public interface IActivationFunction
{
    double Forward(double input);
    double Back(double input);
}