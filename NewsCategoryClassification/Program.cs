using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using NewsCategoryClassification;
using NewsCategoryClassification.Data;
using System.Reflection;
using static Microsoft.ML.DataOperationsCatalog;

internal class Program
{
    static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
    static string _dataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "News_Category_Dataset_v3.json");
    static string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

    static MLContext _mlContext;
    static PredictionEngine<Article, ArticlePrediction> _predEngine;
    static ITransformer _trainedModel;
    static TrainTestData _dataView;

    private static void Main(string[] args)
    {
        _mlContext = new MLContext(seed: 0);

        _dataView = LoadData(_dataPath);

        _trainedModel = BuildAndTrainModel(_mlContext, _dataView.TrainSet);

        Evalute(_mlContext, _trainedModel, _dataView.TestSet);

        UseModelWithSingleItem(_mlContext, _trainedModel);
    }


    static TrainTestData LoadData(string dataPath)
    {
        List<Article> data = JSONextractor.ReadData(dataPath);

        IDataView dataView = _mlContext.Data.LoadFromEnumerable(data);

        TrainTestData splitDataView = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        return splitDataView;
    }

    static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
    {

        //var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
        //    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Headline), outputColumnName: "FeaturizedHeadline"))
        //    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Short_Description), outputColumnName: "FeaturizedShortDescription"))
        //    .Append(mlContext.Transforms.Concatenate("Features", "FeaturizedHeadline", "FeaturizedShortDescription"));

        //pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        //    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Headline), outputColumnName: "FeaturizedHeadline"))
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Short_Description), outputColumnName: "FeaturizedShortDescription"))
            .Append(mlContext.Transforms.Concatenate("Features", "FeaturizedHeadline", "FeaturizedShortDescription"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(splitTrainSet);

        return model;
    }

    private static void Evalute(MLContext mlContext, ITransformer trainedModel, IDataView splitTestSet)
    {
        Console.WriteLine("=============== Evaluating Model accuracy with Test data ===============");
        var predictions = trainedModel.Transform(splitTestSet);

        var testMetrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"*************************************************************************************************************");
        Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
        Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
        Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
        Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
        Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
        Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
        Console.WriteLine($"*************************************************************************************************************");
    }

    private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
    {
        _predEngine = mlContext.Model.CreatePredictionEngine<Article, ArticlePrediction>(model);
        Article article = new Article()
        {
            Link = "https://www.huffpost.com/entry/covid-boosters-uptake-us_n_632d719ee4b087fae6feaac9",
            Headline = "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
            Category = "U.S. NEWS",
            Short_Description = "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.",
            Authors = "Carla K. Johnson, AP",
            Date = "2022-09-23"
        };

        var prediction = _predEngine.Predict(article);

        Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Category} ===============");
    }
}