namespace bpnn_csharp;

public class ReLu : IActivationFunction
{
    public double Forward(double input)
    {
        return input > 0 ? input : 0;
    }
    public double Back(double input)
    {
        return input >= 0 ? 1 : 0;
    }
    public static ReLu Instance = new ReLu();
}


public class Linear : IActivationFunction
{
    public double Forward(double input)
    {
        return input;
    }
    public double Back(double input)
    {
        return input;
    }
    public static Linear Instance = new Linear();
}