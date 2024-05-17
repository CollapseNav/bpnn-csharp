using Collapsenav.Net.Tool;
using Collapsenav.Net.Tool.Excel;
using bpnn_csharp;
NetWork model = new NetWork();
model.AddLayer(5, ReLu.Instance);
model.AddLayer(6, ReLu.Instance);
model.AddLayer(1, Linear.Instance);
Console.WriteLine();

var reader = new MiniExcelReader("./Student_Performance.xlsx");
var datas = new ReadConfig<StudentPreference>()
.Add("Hours Studied", i => i.Hours)
.Add("Previous Scores", i => i.PreScores)
.Add("Extracurricular Activities", i => i.Ext, i => i == "NO" ? 1 : 2)
.Add("Sleep Hours", i => i.Sleep)
.Add("Sample Question Papers Practiced", i => i.Papers)
.Add("Performance Index", i => i.Score)
.ToEntity(reader);

var rate = 0.7;
var train = datas.Take((int)(datas.Count() * rate)).ToList();
var test = datas.Skip(train.Count()).ToList();

// 一些超参设置
var lr = 1e-8;
var epoch = 50000;
var targetError = 1e-3;
var mont = 1;

var count = train.Count / 10;

// 开始训练
Console.WriteLine("Train---------------");
for (; epoch-- > 0;)
{
    var trainError = 0.0;
    double error = 0;
    foreach (var data in train.Shuffle().Take(count))
    {
        model.Forward(data.GetInput());
        model.Back(data.GetOutput(), lr, mont);
        error += model.GetError(data.GetOutput());
    }
    trainError += error / count;
    trainError = Math.Abs(trainError);
    Console.WriteLine(trainError);
    // 达到目标误差时停止训练
    if (trainError < targetError || error == double.NaN)
        break;
}

// 开始测试
Console.WriteLine("Test---------------");
var testError = 0.0;
foreach (var data in test)
{
    var output = model.Forward(data.GetInput());
    Console.WriteLine($"{data.GetOutput()[0].PadRight(10)}{output.First()}");
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