using Collapsenav.Net.Tool;
using Collapsenav.Net.Tool.Excel;
using bpnn_csharp;

var reader = new MiniExcelReader("./Student_Performance.xlsx");
var datas = new ReadConfig<StudentPreference>()
.Add("Hours Studied", i => i.Hours)
.Add("Previous Scores", i => i.PreScores)
.Add("Extracurricular Activities", i => i.Ext, i => i == "NO" ? 1 : 2)
.Add("Sleep Hours", i => i.Sleep)
.Add("Sample Question Papers Practiced", i => i.Papers)
.Add("Performance Index", i => i.Score)
.ToEntity(reader);

// 划分训练集和测试集, 70%训练集，30%测试集
var rate = 0.7;
var train = datas.Take((int)(datas.Count() * rate)).ToList();
var test = datas.Skip(train.Count()).ToList();

// 一些超参设置
// 学习率
var lr = 1e-8;
// 最大迭代次数
var maxepoch = 50000;
// 目标误差
var targetError = 1e-3;
// 随机数种子
var seed = 0;
var mont = 1;

// 减少运算量, 每次只取十分之一进行计算
var count = train.Count / 10;


// 构建网络并且设置随机种子
NetWork model = new NetWork();
model.AddLayer(5, ReLu.Instance)
.AddLayer(6, ReLu.Instance)
.AddLayer(1, Linear.Instance)
.SetRandSeed(seed);

// 开始训练
Console.WriteLine("Train---------------");
var trainError = 0.0;
for (int epoch = 0; epoch++ < maxepoch;)
{
    double error = 0;
    // 随机取十分之一的训练数据进行训练
    foreach (var data in train.Shuffle().Take(count))
    {
        // 前向传播计算输出
        model.Forward(data.GetInput());
        // 反向传播
        model.Back(data.GetOutput(), lr, mont);
        // 累计误差
        error += model.GetError(data.GetOutput());
    }
    trainError += error / count;
    // 每10次输出一次误差
    if (epoch % 10 == 0)
    {
        trainError = Math.Abs(trainError) / 10;
        Console.WriteLine($"{epoch / 10}-epoch:{epoch}, error:{trainError}");
        // 达到目标误差时停止训练
        if (trainError < targetError || error == double.NaN)
            break;
        trainError = 0;
    }
}

// 开始测试
Console.WriteLine("Test---------------");
var testError = 0.0;
foreach (var data in test)
{
    // 前向传播计算输出
    var output = model.Forward(data.GetInput());
    // 输出真值和预测值进行对比
    Console.WriteLine($"{data.GetOutput()[0].PadRight(10)}{output.First()}");
    // 累计误差
    testError += model.GetError(data.GetOutput());
}
// 输出平均测试误差
Console.WriteLine(testError / test.Count);

class StudentPreference
{
    /// <summary>
    /// 每个学生学习的总小时数
    /// </summary>
    public double Hours { get; set; }
    /// <summary>
    /// 学生在以前的测试中获得的分数
    /// </summary>
    public double PreScores { get; set; }
    /// <summary>
    /// 学生是否参加课外活动（是或否）
    /// </summary>
    public double Ext { get; set; }
    /// <summary>
    /// 学生每天的平均睡眠小时数
    /// </summary>
    public double Sleep { get; set; }
    /// <summary>
    /// 学生练习的样题数量
    /// </summary>
    public double Papers { get; set; }
    /// <summary>
    /// 衡量每个学生的整体表现
    /// </summary>
    public double Score { get; set; }
    public double[] GetInput()
    {
        return new double[] { Hours, PreScores, Ext, Sleep, Papers };
    }
    public double[] GetOutput()
    {
        return new double[] { Score };
    }
}