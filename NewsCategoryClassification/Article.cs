using Microsoft.ML.Data;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewsCategoryClassification
{
    public class Article
    {
        [JsonProperty("link")]
        [ColumnName("Link")]
        public string Link;
        [JsonProperty("headline")]
        [ColumnName("Headline")]
        public string Headline;
        [JsonProperty("category")]
        [ColumnName("Category")]
        public string Category;
        [JsonProperty("short_description")]
        [ColumnName("Short_Description")]
        public string Short_Description;
        [JsonProperty("authors")]
        [ColumnName("Authors")]
        public string Authors;
        [JsonProperty("date")]
        [ColumnName("Date")]
        public string Date;
    }

    public class ArticlePrediction
    {
        [ColumnName("PredictedLabel")]
        public string? Category;

        //[ColumnName("Score")]
        //public float Score { get; set; }
    }
}
