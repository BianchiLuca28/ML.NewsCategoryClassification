using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace NewsCategoryClassification.Data
{
    static public class JSONextractor
    {
        public static List<Article> ReadData(string dataPath)
        {
            List<Article> articles = new List<Article>();

            using (StreamReader r = new StreamReader(dataPath))
            {
                string json = r.ReadToEnd();

                articles = new List<Article>();

                JsonTextReader reader = new JsonTextReader(new StringReader(json));
                reader.SupportMultipleContent = true;

                while (true)
                {
                    if (!reader.Read())
                    {
                        break;
                    }

                    JsonSerializer serializer = new JsonSerializer();
                    Article article = serializer.Deserialize<Article>(reader);

                    articles.Add(article);
                }
            }

            return articles;
        }
    }
}
