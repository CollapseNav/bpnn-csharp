namespace bpnn_csharp;

public class ReLu : IActivationFunction
{
    public double Forward(double input) => input > 0 ? input : 0;
    public double Back(double input) => input >= 0 ? 1 : 0;
    public static ReLu Instance = new ReLu();
}
public class Linear : IActivationFunction
{
    public double Forward(double input) => input;
    public double Back(double input) => input;
    public static Linear Instance = new Linear();
}