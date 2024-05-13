using Collapsenav.Net.Tool;
using Collapsenav.Net.Tool.Excel;
using bpnn_csharp;
NetWork model = new NetWork();
model.AddLayer(5, ReLu.Instance);
model.AddLayer(8, ReLu.Instance);
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
.ToEntity(reader).Shuffle();
var rate = 0.7;
var train = datas.Take((int)(datas.Count() * rate)).ToList();
var test = datas.Skip(train.Count()).ToList();
var lr = 1e-8;
var epoch = 50000;
var targetError = 1e-3;
var mont = 1;

Console.WriteLine("Test---------------");
for (; epoch-- > 0;)
{
    double error = 0;
    foreach (var data in train.Shuffle(3))
    {
        model.Forward(data.GetInput());
        model.Back(data.GetOutput(), lr, mont);
        error = Math.Abs(model.GetError(data.GetOutput()[0]));
    }
    if (epoch % 10 == 0)
        Console.WriteLine(error);
    if (error < targetError || error == double.NaN)
        break;
}

Console.WriteLine("Test---------------");
var testError = 0.0;
foreach (var data in test)
{
    model.Forward(data.GetInput());
    Console.WriteLine($"{data.GetOutput()[0].PadRight(10)}{model.Output.First().Value}");
    testError += model.GetError(data.GetOutput()[0]);
}
Console.WriteLine(testError / test.Count);

class Data
{
    public double SAT { get; set; }
    public double GPA { get; set; }
}

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