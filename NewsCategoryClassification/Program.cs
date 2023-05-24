using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using NewsCategoryClassification;
using NewsCategoryClassification.Data;
using System.Reflection;
using static Microsoft.ML.DataOperationsCatalog;
using MathNet.Numerics.Statistics;

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

        EvaluteTrainData(_mlContext, _trainedModel, _dataView.TrainSet);

        EvaluteTestData(_mlContext, _trainedModel, _dataView.TestSet);

        UseModelWithSingleItem(_mlContext, _trainedModel);
    }


    static TrainTestData LoadData(string dataPath)
    {
        List<Article> data = JSONextractor.ReadData(dataPath);

        DataStatistics(data);

        IDataView dataView = _mlContext.Data.LoadFromEnumerable(data);

        TrainTestData splitDataView = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        return splitDataView;
    }

    private static void DataStatistics(List<Article> data)
    {
        var maxCategoryName = "";
        var maxCategoryMultiplicity = -1;

        var minCategoryName = "";
        var minCategoryMultiplicity = 50000;

        var count_categories = data.GroupBy(x => x.Category).Count();

        var categories_multiplicities = data.GroupBy(x => x.Category).Select(category => category.Count());

        var categories = data.GroupBy(x => x.Category);

        Console.WriteLine($"======= Il numero di Categorie sono: {count_categories} ===========");

        Console.WriteLine("********************************************************************");
        Console.WriteLine("****************Printing Categories Multiplicity********************");

        foreach (var category in categories)
        {
            Console.WriteLine($" - Category: {category.Key} , Multiplicity: {category.Count()}");
            if (category.Count() > maxCategoryMultiplicity)
            {
                maxCategoryMultiplicity = category.Count();
                maxCategoryName = category.Key;
            } else if (category.Count() < minCategoryMultiplicity)
            {
                minCategoryMultiplicity = category.Count();
                minCategoryName = category.Key;
            }
        }
        Console.WriteLine("********************************************************************");

        Console.WriteLine($" - MAX Category: {maxCategoryName} , Count: {maxCategoryMultiplicity}");
        Console.WriteLine($" - Min Category: {minCategoryName} , Count: {minCategoryMultiplicity}");

        Console.WriteLine("********************************************************************");
        Console.WriteLine("***************************STATISTICS*******************************");
        var mean = categories_multiplicities.Sum() / (double) categories_multiplicities.Count();
        Console.WriteLine($" - Mean/Average: {mean}");
        // Console.WriteLine($" - Variance: {(categories_multiplicities.Select(x => (x - mean)).Sum()) / categories_multiplicities.Count()}");

        Console.WriteLine("********************************************************************");
    }

    private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
    {

        //IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
        //    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Headline), outputColumnName: "FeaturizedHeadline"))
        //    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Short_Description), outputColumnName: "FeaturizedShortDescription"))
        //    .Append(mlContext.Transforms.Concatenate("Features", "FeaturizedHeadline", "FeaturizedShortDescription"));

        //var training_pipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        //    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Headline), outputColumnName: "FeaturizedHeadline"))
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Short_Description), outputColumnName: "FeaturizedShortDescription"))
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Authors), outputColumnName: "EncodedAuthors"))
            .Append(mlContext.Transforms.Concatenate("Features", "FeaturizedHeadline", "FeaturizedShortDescription", "EncodedAuthors"))
            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(splitTrainSet);

        return model;
    }

    private static void EvaluteTrainData(MLContext mlContext, ITransformer trainedModel, IDataView splitTrainSet)
    {
        Console.WriteLine("=============== Evaluating Model accuracy with Train data ===============");
        var predictions = trainedModel.Transform(splitTrainSet);

        var testMetrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"*************************************************************************************************************");
        Console.WriteLine($"*       Metrics for Multi-class Classification model - Train Data     ");
        Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
        Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
        Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
        Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
        Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
        Console.WriteLine($"*************************************************************************************************************");
    }

    private static void EvaluteTestData(MLContext mlContext, ITransformer trainedModel, IDataView splitTestSet)
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
        Article article1 = new Article()
        {
            Link = "https://www.huffpost.com/entry/covid-boosters-uptake-us_n_632d719ee4b087fae6feaac9",
            Headline = "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
            Category = "U.S. NEWS",
            Short_Description = "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.",
            Authors = "Carla K. Johnson, AP",
            Date = "2022-09-23"
        };

        var prediction1 = _predEngine.Predict(article1);

        Console.WriteLine($"=============== FIRST Single Prediction just-trained-model - Result: {prediction1.Category} ===============");

        Article article2 = new Article()
        {
            Link = "https://www.huffpost.com/entry/american-airlines-passenger-banned-flight-attendant-punch-justice-department_n_632e25d3e4b0e247890329fe",
            Headline = "American Airlines Flyer Charged, Banned For Life After Punching Flight Attendant On Video",
            Category = "U.S. NEWS",
            Short_Description = "He was subdued by passengers and crew when he fled to the back of the aircraft after the confrontation, according to the U.S. attorney's office in Los Angeles.",
            Authors = "Mary Papenfuss",
            Date = "2022-09-23"
        };

        var prediction2 = _predEngine.Predict(article2);

        Console.WriteLine($"=============== SECOND (False) Single Prediction just-trained-model - Result: {prediction2.Category} ===============");

        Article article3 = new Article()
        {
            Link = "https://www.huffpost.com/entry/american-airlines-passenger-banned-flight-attendant-punch-justice-department_n_632e25d3e4b0e247890329fe",
            Headline = "Citing Imminent Danger Cloudflare Drops Hate Site Kiwi Farms",
            Category = "TECH",
            Short_Description = "Cloudflare CEO Matthew Prince had previously resisted calls to block the site.",
            Authors = "The Associated Press, AP",
            Date = "2022-09-05"
        };

        var prediction3 = _predEngine.Predict(article3);

        Console.WriteLine($"=============== THIRD (False) Single Prediction just-trained-model - Result: {prediction3.Category} ===============");
    }
}