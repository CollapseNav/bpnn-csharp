namespace bpnn_csharp;

/// <summary>
/// 激活函数
/// </summary>
public interface IActivationFunction
{
    /// <summary>
    /// 前向
    /// </summary>
    double Forward(double input);
    /// <summary>
    /// 反向
    /// </summary>
    double Back(double input);
}