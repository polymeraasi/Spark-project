// Databricks notebook source
// MAGIC # Data-Intensive Programming - Group assignment

// Authors: Pinja Koivisto & Janina Montonen

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType, DateType, DoubleType, IntegerType}
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator


// MAGIC %md
// MAGIC ## Basic Task 1 - Video game sales data
// MAGIC
// MAGIC The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container]
// MAGIC
// MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
// MAGIC
// MAGIC Using the data, find answers to the following:
// MAGIC
// MAGIC - Which publisher has the highest total sales in video games in North America considering games released in years 2006-2015?
// MAGIC - How many titles in total for this publisher do not have sales data available for North America considering games released in years 2006-2015?
// MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2006-2015, what are the total sales, in North America and globally, for each year?
// MAGIC     - I.e., what are the total sales (in North America and globally) for games released by this publisher in year 2006? And the same for year 2007? ...
// MAGIC


// reading the csv file into a dataframe
val path = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv"

val videoGameSchema: StructType = new StructType(
    Array(
        StructField("title", StringType, true),
        StructField("publisher", StringType, true),
        StructField("developer", StringType, true),
        StructField("release_date", DateType, true),
        StructField("platform", StringType, true),
        StructField("total_sales", DoubleType, true),
        StructField("na_sales", DoubleType, true),
        StructField("japan_sales", DoubleType, true),
        StructField("pal_sales", DoubleType, true),
        StructField("other_sales", DoubleType, true),
        StructField("user_score", DoubleType, true),
        StructField("critic_score", DoubleType, true)
    )
)

val videoGameDF: DataFrame = spark.read.option("header", "true")
    .option("sep", "|")
    .schema(videoGameSchema)
    .csv(path)

// only selecting columns we need
val videoGameDFSelected : DataFrame = videoGameDF.select("publisher", "release_date", "total_sales", "na_sales")

// creating a new column for year from release_date
val videoGameDFWithYear: DataFrame = videoGameDFSelected.withColumn("year", year(col("release_date")))

// ###### FIRST QUESTION ####### //
// Which publisher has the highest total sales in video games in North America considering games released in years 2006-2015?

// calculating total sales for each publisher
val publisherSalesDF: DataFrame = videoGameDFWithYear.filter((col("year") >= 2006) && (col("year") <= 2015))
    .groupBy("publisher")
    .agg(sum("na_sales").alias("total_sales"))

// finding the maximum of the total sales
val maxTotalSales : Double = publisherSalesDF.agg(max("total_sales").alias("max_total_sales")).collect()(0)(0).asInstanceOf[Double]

// finding the publisher with the maximum total sales
val bestNAPublisher: String = publisherSalesDF.filter(col("total_sales") === maxTotalSales).select("publisher").first().getString(0)

// ####### SECOND QUESTION ####### //
// How many titles in total for this publisher do not have sales data available for North America considering games released in years 2006-2015?

// filtering the videoGameDFSelected with the publisher from Q1
val titlesWithMissingSalesData: Long = videoGameDFWithYear.filter((col("publisher") === bestNAPublisher) && (col("year") >= 2006) && (col("year") <= 2015) && (col("na_sales").isNull)).count()

// ####### THIRD QUESTION ######## //

// same filters (publisher, year range), now fetching total sales across the globe for each year in the range
val bestNAPublisherSales: DataFrame = videoGameDFWithYear.filter((col("publisher") === bestNAPublisher) && (col("year") >= 2006) && (col("year") <= 2015))
    .groupBy("year")
    .agg(round(sum("na_sales"), 2).alias("na_total"), round(sum("total_sales"), 2).alias("global_total"))
    .orderBy("year")

println(s"The publisher with the highest total video game sales in North America is: '$bestNAPublisher'")
println(s"The number of titles with missing sales data for North America: $titlesWithMissingSalesData")
println("Sales data for the publisher:")
bestNAPublisherSales.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 2 - Event data from football matches
// MAGIC
// MAGIC A parquet file in the [Shared container] based on [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1.
// MAGIC
// MAGIC #### Background information
// MAGIC
// MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
// MAGIC
// MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
// MAGIC
// MAGIC The team that scores more goals than their opponent wins the match.
// MAGIC
// MAGIC **Columns in the data**
// MAGIC
// MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.
// MAGIC
// MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
// MAGIC
// MAGIC | column name | column type | description |
// MAGIC | ----------- | ----------- | ----------- |
// MAGIC | competition | string | The name of the competition |
// MAGIC | season | string | The season the match was played |
// MAGIC | matchId | integer | A unique id for the match |
// MAGIC | eventId | integer | A unique id for the event |
// MAGIC | homeTeam | string | The name of the home team |
// MAGIC | awayTeam | string | The name of the away team |
// MAGIC | event | string | The main category for the event |
// MAGIC | subEvent | string | The subcategory for the event |
// MAGIC | eventTeam | string | The name of the team that initiated the event |
// MAGIC | eventPlayerId | integer | The id for the player who initiated the event |
// MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
// MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
// MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
// MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
// MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
// MAGIC
// MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
// MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
// MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
// MAGIC
// MAGIC #### The task
// MAGIC
// MAGIC In this task you should load the data with all the rows into a data frame. This data frame object will then be used in the following basic tasks 3-8.

// COMMAND ----------

val readEventDF: DataFrame = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/events.parquet")
val eventDF : DataFrame = readEventDF.select("competition", "season", "matchId", "homeTeam", "awayTeam", "event", "eventTeam", "tags").cache()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 3 - Calculate match results
// MAGIC
// MAGIC Create a match data frame for all the matches included in the event data frame created in basic task 2.
// MAGIC
// MAGIC The resulting data frame should contain one row for each match and include the following columns:
// MAGIC
// MAGIC | column name   | column type | description |
// MAGIC | ------------- | ----------- | ----------- |
// MAGIC | matchId       | integer     | A unique id for the match |
// MAGIC | competition   | string      | The name of the competition |
// MAGIC | season        | string      | The season the match was played |
// MAGIC | homeTeam      | string      | The name of the home team |
// MAGIC | awayTeam      | string      | The name of the away team |
// MAGIC | homeTeamGoals | integer     | The number of goals scored by the home team |
// MAGIC | awayTeamGoals | integer     | The number of goals scored by the away team |
// MAGIC
// MAGIC The number of goals scored for each team should be determined by the available event data.<br>
// MAGIC There are two events related to each goal:
// MAGIC
// MAGIC - One event for the player that scored the goal. This includes possible own goals.
// MAGIC - One event for the goalkeeper that tried to stop the goal.
// MAGIC
// MAGIC You need to choose which types of events you are counting.<br>
// MAGIC If you count both of the event types mentioned above, you will get double the amount of actual goals.

// COMMAND ----------

// calculating goals and own goals
val goalsDF: DataFrame = eventDF.withColumn("Goal", 
    when(array_contains(col("tags"), "Goal") && array_contains(col("tags"), "Accurate"), 1).otherwise(0))
  .withColumn("OwnGoal", 
    when(array_contains(col("tags"), "Own goal"), 1).otherwise(0))

// assigning goals right
val goalsAssignedDF: DataFrame = goalsDF.withColumn("homeTeamGoal", 
    when((col("eventTeam") === col("homeTeam")) && (col("Goal") === 1) && (col("OwnGoal") === 0), 1)
    .when((col("eventTeam") === col("awayTeam")) && (col("OwnGoal") === 1), 1).otherwise(0))
  .withColumn("awayTeamGoal", 
    when((col("eventTeam") === col("awayTeam")) && (col("Goal") === 1) && (col("OwnGoal") === 0), 1)
    .when((col("eventTeam") === col("homeTeam")) && (col("OwnGoal") === 1), 1).otherwise(0))

val matchDF: DataFrame = goalsAssignedDF.groupBy("matchId", "competition", "season", "homeTeam", "awayTeam").agg(
    sum(col("homeTeamGoal")).alias("homeTeamGoals"),
    sum(col("awayTeamGoal")).alias("awayTeamGoals")
).cache()


// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 4 - Calculate team points in a season
// MAGIC
// MAGIC Create a season data frame that uses the match data frame from the basic task 3 and contains aggregated seasonal results and statistics for all the teams in all leagues. While the used dataset only includes data from a single season for each league, the code should be written such that it would work even if the data would include matches from multiple seasons for each league.
// MAGIC
// MAGIC ###### Game result determination
// MAGIC
// MAGIC - Team wins the match if they score more goals than their opponent.
// MAGIC - The match is considered a draw if both teams score equal amount of goals.
// MAGIC - Team loses the match if they score fewer goals than their opponent.
// MAGIC
// MAGIC ###### Match point determination
// MAGIC
// MAGIC - The winning team gains 3 points from the match.
// MAGIC - Both teams gain 1 point from a drawn match.
// MAGIC - The losing team does not gain any points from the match.
// MAGIC
// MAGIC The resulting data frame should contain one row for each team per league and season. It should include the following columns:
// MAGIC
// MAGIC | column name    | column type | description |
// MAGIC | -------------- | ----------- | ----------- |
// MAGIC | competition    | string      | The name of the competition |
// MAGIC | season         | string      | The season |
// MAGIC | team           | string      | The name of the team |
// MAGIC | games          | integer     | The number of games the team played in the given season |
// MAGIC | wins           | integer     | The number of wins the team had in the given season |
// MAGIC | draws          | integer     | The number of draws the team had in the given season |
// MAGIC | losses         | integer     | The number of losses the team had in the given season |
// MAGIC | goalsScored    | integer     | The total number of goals the team scored in the given season |
// MAGIC | goalsConceded  | integer     | The total number of goals scored against the team in the given season |
// MAGIC | points         | integer     | The total number of points gained by the team in the given season |

// COMMAND ----------

val teamsDF: DataFrame = matchDF
  .select(
    col("competition"),
    col("season"),
    col("homeTeam").alias("team"),
    col("homeTeamGoals").alias("goalsScored"),
    col("awayTeamGoals").alias("goalsConceded"),
    when(col("homeTeamGoals") > col("awayTeamGoals"), 3) 
      .when(col("homeTeamGoals") === col("awayTeamGoals"), 1)
      .otherwise(0).alias("points"),
    lit(1).alias("games"), 
    when(col("homeTeamGoals") > col("awayTeamGoals"), 1).otherwise(0).alias("wins"), 
    when(col("homeTeamGoals") === col("awayTeamGoals"), 1).otherwise(0).alias("draws"), 
    when(col("homeTeamGoals") < col("awayTeamGoals"), 1).otherwise(0).alias("losses") 
  )
  .union(
    matchDF.select(
      col("competition"),
      col("season"),
      col("awayTeam").alias("team"),
      col("awayTeamGoals").alias("goalsScored"),
      col("homeTeamGoals").alias("goalsConceded"),
      when(col("awayTeamGoals") > col("homeTeamGoals"), 3)
        .when(col("awayTeamGoals") === col("homeTeamGoals"), 1)
        .otherwise(0).alias("points"),
      lit(1).alias("games"),
      when(col("awayTeamGoals") > col("homeTeamGoals"), 1).otherwise(0).alias("wins"),
      when(col("awayTeamGoals") === col("homeTeamGoals"), 1).otherwise(0).alias("draws"),
      when(col("awayTeamGoals") < col("homeTeamGoals"), 1).otherwise(0).alias("losses")
    )
  )

val seasonDF: DataFrame = teamsDF.groupBy("competition", "season", "team")
  .agg(
    sum("games").alias("games"),
    sum("wins").alias("wins"),
    sum("draws").alias("draws"),
    sum("losses").alias("losses"),
    sum("goalsScored").alias("goalsScored"),
    sum("goalsConceded").alias("goalsConceded"),
    sum("points").alias("points")
  )
  .orderBy("season", "competition", "team").cache()


matchDF.unpersist()


// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 5 - English Premier League table
// MAGIC
// MAGIC Using the season data frame from basic task 4 calculate the final league table for `English Premier League` in season `2017-2018`.
// MAGIC
// MAGIC The result should be given as data frame which is ordered by the team's classification for the season.
// MAGIC
// MAGIC A team is classified higher than the other team if one of the following is true:
// MAGIC
// MAGIC - The team has a higher number of total points than the other team
// MAGIC - The team has an equal number of points, but have a better goal difference than the other team
// MAGIC - The team has an equal number of points and goal difference, but have more goals scored in total than the other team
// MAGIC
// MAGIC Goal difference is the difference between the number of goals scored for and against the team.
// MAGIC
// MAGIC The resulting data frame should contain one row for each team.<br>
// MAGIC It should include the following columns (several columns renamed trying to match the [league table in Wikipedia](https://en.wikipedia.org/wiki/2017%E2%80%9318_Premier_League#League_table)):
// MAGIC
// MAGIC | column name | column type | description |
// MAGIC | ----------- | ----------- | ----------- |
// MAGIC | Pos         | integer     | The classification of the team |
// MAGIC | Team        | string      | The name of the team |
// MAGIC | Pld         | integer     | The number of games played |
// MAGIC | W           | integer     | The number of wins |
// MAGIC | D           | integer     | The number of draws |
// MAGIC | L           | integer     | The number of losses |
// MAGIC | GF          | integer     | The total number of goals scored by the team |
// MAGIC | GA          | integer     | The total number of goals scored against the team |
// MAGIC | GD          | string      | The goal difference |
// MAGIC | Pts         | integer     | The total number of points gained by the team |
// MAGIC
// MAGIC The goal difference should be given as a string with an added `+` at the beginning if the difference is positive, similarly to the table in the linked Wikipedia article.

// COMMAND ----------

val englandSeasonDF: DataFrame = seasonDF
  .filter(col("competition") === "English Premier League" && col("season") === "2017-2018")

val englandWithGD: DataFrame = englandSeasonDF
  .withColumn("GD", 
    when(col("goalsScored") - col("goalsConceded") > 0, 
         concat(lit("+"), (col("goalsScored") - col("goalsConceded")).cast("string")))
    .otherwise((col("goalsScored") - col("goalsConceded")).cast("string"))
  )

val englandOrdered: DataFrame = englandWithGD
  .orderBy(
    col("points").desc,
    (col("goalsScored") - col("goalsConceded")).desc,
    col("goalsScored").desc
  )

val englandWithPosition: DataFrame = englandOrdered
  .withColumn("Pos", row_number().over(Window.orderBy(
    col("points").desc,
    (col("goalsScored") - col("goalsConceded")).desc,
    col("goalsScored").desc
  )))

val englandDF = englandWithPosition.select(
  col("Pos"),
  col("team").alias("Team"),
  col("games").alias("Pld"),
  col("wins").alias("W"),
  col("draws").alias("D"),
  col("losses").alias("L"),
  col("goalsScored").alias("GF"),
  col("goalsConceded").alias("GA"),
  col("GD"),
  col("points").alias("Pts")
)

println("English Premier League table for season 2017-2018")
englandDF.show(20, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic task 6: Calculate the number of passes
// MAGIC
// MAGIC This task involves going back to the event data frame and counting the number of passes each team made in each match. A pass is considered successful if it is marked as `Accurate`.
// MAGIC
// MAGIC Using the event data frame from basic task 2, calculate the total number of passes as well as the total number of successful passes for each team in each match.<br>
// MAGIC The resulting data frame should contain one row for each team in each match, i.e., two rows for each match. It should include the following columns:
// MAGIC
// MAGIC | column name | column type | description |
// MAGIC | ----------- | ----------- | ----------- |
// MAGIC | matchId     | integer     | A unique id for the match |
// MAGIC | competition | string      | The name of the competition |
// MAGIC | season      | string      | The season |
// MAGIC | team        | string      | The name of the team |
// MAGIC | totalPasses | integer     | The total number of passes the team attempted in the match |
// MAGIC | successfulPasses | integer | The total number of successful passes made by the team in the match |
// MAGIC
// MAGIC You can assume that each team had at least one pass attempt in each match they played.

// COMMAND ----------

val passesDF: DataFrame = eventDF.withColumn("totalPasses", when(col("event") === "Pass", 1).otherwise(0)).withColumn("successfulPasses", when(array_contains(col("tags"), "Accurate") && col("event") === "Pass", 1).otherwise(0))

val passDF: DataFrame = passesDF.groupBy("matchId", "eventTeam", "competition", "season").agg(
  sum(col("successfulPasses")).as("successfulPasses"),
  sum(col("totalPasses")).as("totalPasses")
)

val matchPassDF: DataFrame = passDF.select(
  col("matchId"),
  col("eventTeam").as("team"),
  col("competition"),
  col("season"),
  col("successfulPasses"),
  col("totalPasses")
).cache()

eventDF.unpersist()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 7: Teams with the worst passes
// MAGIC
// MAGIC Using the match pass data frame from basic task 6 find the teams with the lowest average ratio for successful passes over the season `2017-2018` for each league.
// MAGIC
// MAGIC The ratio for successful passes over a single match is the number of successful passes divided by the number of total passes.<br>
// MAGIC The average ratio over the season is the average of the single match ratios.
// MAGIC
// MAGIC Give the result as a data frame that has one row for each league-team pair with the following columns:
// MAGIC
// MAGIC | column name | column type | description |
// MAGIC | ----------- | ----------- | ----------- |
// MAGIC | competition | string      | The name of the competition |
// MAGIC | team        | string      | The name of the team |
// MAGIC | passSuccessRatio | double | The average ratio for successful passes over the season given as percentages rounded to two decimals |
// MAGIC
// MAGIC Order the data frame so that the team with the lowest ratio for passes is given first.

// COMMAND ----------

val lowestPassSuccessRatioDF: DataFrame = matchPassDF.where(col("season") === "2017-2018")
.groupBy("competition", "team")
.agg(
  round(avg((col("successfulPasses") / (col("totalPasses")) * 100)), 2).as("passSuccessRatio")
)
.orderBy("passSuccessRatio").cache()

println("The teams with the lowest ratios for successful passes for each league in season 2017-2018:")
lowestPassSuccessRatioDF.show(5, false)

matchPassDF.unpersist()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic task 8: The best teams
// MAGIC
// MAGIC For this task the best teams are determined by having the highest point average per match.
// MAGIC
// MAGIC Using the data frames created in the previous tasks find the two best teams from each league in season `2017-2018` with their full statistics.
// MAGIC
// MAGIC Give the result as a data frame with the following columns:
// MAGIC
// MAGIC | column name | column type | description |
// MAGIC | ----------- | ----------- | ----------- |
// MAGIC | Team        | string      | The name of the team |
// MAGIC | League      | string      | The name of the league |
// MAGIC | Pos         | integer     | The classification of the team within their league |
// MAGIC | Pld         | integer     | The number of games played |
// MAGIC | W           | integer     | The number of wins |
// MAGIC | D           | integer     | The number of draws |
// MAGIC | L           | integer     | The number of losses |
// MAGIC | GF          | integer     | The total number of goals scored by the team |
// MAGIC | GA          | integer     | The total number of goals scored against the team |
// MAGIC | GD          | string      | The goal difference |
// MAGIC | Pts         | integer     | The total number of points gained by the team |
// MAGIC | Avg         | double      | The average points per match gained by the team |
// MAGIC | PassRatio   | double      | The average ratio for successful passes over the season given as percentages rounded to two decimals |
// MAGIC
// MAGIC Order the data frame so that the team with the highest point average per match is given first.

// COMMAND ----------


val season2017DF: DataFrame = seasonDF.filter(col("season") === "2017-2018")

val avgPointsDF: DataFrame = season2017DF
  .withColumn("Avg", round(col("points") / col("games"), 2)).withColumn("GD", 
    when(col("goalsScored") - col("goalsConceded") > 0, 
         concat(lit("+"), (col("goalsScored") - col("goalsConceded")).cast("string")))
    .otherwise((col("goalsScored") - col("goalsConceded")).cast("string")))

val seasonWithPassRatioDF: DataFrame = avgPointsDF
  .join(
    lowestPassSuccessRatioDF.withColumnRenamed("passSuccessRatio", "PassRatio"),
    Seq("competition", "team"),
    "left"
  )

val rankedSeasonDF: DataFrame = seasonWithPassRatioDF
  .withColumn("Pos", row_number().over(Window.partitionBy("competition").orderBy(col("points").desc,
    (col("goalsScored") - col("goalsConceded")).desc,
    col("goalsScored").desc)))

val bestDF: DataFrame = rankedSeasonDF
  .filter(col("Pos") <= 2)
  .select(
    col("team").as("Team"),
    col("competition"),
    col("Pos"),
    col("games").as("Pld"),
    col("wins").as("W"),
    col("draws").as("D"),
    col("losses").as("L"),
    col("goalsScored").as("GF"),
    col("goalsConceded").as("GA"),
    col("GD"),
    col("points").as("Pts"),
    col("Avg"),
    col("PassRatio")
  )
  .orderBy(desc("Avg"))

println("The top 2 teams for each league in season 2017-2018")
bestDF.show(10, false)

seasonDF.unpersist()
lowestPassSuccessRatioDF.unpersist()


// MAGIC %md
// MAGIC ## Advanced Task - Machine learning tasks 
// MAGIC
// MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the [ProCem](https://www.senecc.fi/projects/procem-2) research project is used as the training and test data. Similar data in a slightly different format was used in the first tasks of weekly exercise 3.
// MAGIC
// MAGIC The folder `assignment/energy/procem_13m.parquet` in the [Shared container]contains the time series data in Parquet format.
// MAGIC
// MAGIC #### Data description
// MAGIC
// MAGIC The dataset contains time series data from a period of 13 months (from the beginning of May 2023 to the end of May 2024). Each row contains the average of the measured values for a single minute. The following columns are included in the data:
// MAGIC
// MAGIC | column name        | column type   | description |
// MAGIC | ------------------ | ------------- | ----------- |
// MAGIC | time               | long          | The UNIX timestamp in second precision |
// MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
// MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
// MAGIC | wind_speed         | double        | The wind speed measured by the weather station on top of Sähkötalo (`m/s`) |
// MAGIC | power_tenants      | double        | The total combined electricity power used by the tenants on Kampusareena (`W`) |
// MAGIC | power_maintenance  | double        | The total combined electricity power used by the building maintenance systems on Kampusareena (`W`) |
// MAGIC | power_solar_panels | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
// MAGIC | electricity_price  | double        | The market price for electricity in Finland (`€/MWh`) |
// MAGIC
// MAGIC There are some missing values that need to be removed before using the data for training or testing. However, only the minimal amount of rows should be removed for each test case.
// MAGIC
// MAGIC ### Tasks
// MAGIC
// MAGIC - The main task is to train and test a machine learning model with [Random forest classifier](https://spark.apache.org/docs/3.5.0/ml-classification-regression.html#random-forests) in six different cases:
// MAGIC     - Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input
// MAGIC     - Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input
// MAGIC     - Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input
// MAGIC     - Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
// MAGIC     - Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
// MAGIC     - Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
// MAGIC - For each of the six case you are asked to:
// MAGIC     1. Clean the source dataset from rows with missing values.
// MAGIC     2. Split the dataset into training and test parts.
// MAGIC     3. Train the ML model using a Random forest classifier with case-specific input and prediction.
// MAGIC     4. Evaluate the accuracy of the model with Spark built-in multiclass classification evaluator.
// MAGIC     5. Further evaluate the accuracy of the model with a custom build evaluator which should do the following:
// MAGIC         - calculate the percentage of correct predictions
// MAGIC             - this should correspond to the accuracy value from the built-in accuracy evaluator
// MAGIC         - calculate the percentage of predictions that were at most one away from the correct predictions taking into account the cyclic nature of the month and hour values:
// MAGIC             - if the correct month value was `5`, then acceptable predictions would be `4`, `5`, or `6`
// MAGIC             - if the correct month value was `1`, then acceptable predictions would be `12`, `1`, or `2`
// MAGIC             - if the correct month value was `12`, then acceptable predictions would be `11`, `12`, or `1`
// MAGIC         - calculate the percentage of predictions that were at most two away from the correct predictions taking into account the cyclic nature of the month and hour values:
// MAGIC             - if the correct month value was `5`, then acceptable predictions would be from `3` to `7`
// MAGIC             - if the correct month value was `1`, then acceptable predictions would be from `11` to `12` and from `1` to `3`
// MAGIC             - if the correct month value was `12`, then acceptable predictions would be from `10` to `12` and from `1` to `2`
// MAGIC         - calculate the average probability the model predicts for the correct value
// MAGIC             - the probabilities for a single prediction can be found from the `probability` column after the predictions have been made with the model
// MAGIC - As the final part of this advanced task, you are asked to do the same experiments (training+evaluation) with two further cases of your own choosing:
// MAGIC     - you can decide on the input columns yourself
// MAGIC     - you can decide the predicted attribute yourself
// MAGIC     - you can try some other classifier other than the random forest one if you want
// MAGIC
// MAGIC In all cases you are free to choose the training parameters as you wish.<br>
// MAGIC Note that it is advisable that while you are building your task code to only use a portion of the full 13-month dataset in the initial experiments.

// COMMAND ----------

val dataPath = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/energy/procem_13m.parquet"
val df_raw = spark.read.parquet(dataPath)

// removing rows with missing values
val df_cleaned = df_raw.na.drop()

// adding columns month and hour
val df_with_labels = df_cleaned
  .withColumn("month", month(from_unixtime(col("time"))))
  .withColumn("hour", hour(from_unixtime(col("time"))))

// features - 3 sets (each for month, hour)
val weatherFeatures1 = Array("temperature", "humidity", "wind_speed") // temperature, humidity, and wind speed
val weatherFeatures2 = Array("power_tenants", "power_maintenance", "power_solar_panels") // tenants, maintenance, and solar panels
val weatherFeatures3 = Array("temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price")  // weather values, power values, and price

////////////////////////////////////////////////// CASES //////////////////////////////////////////////////////////////////

// case 1: Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input
val df_features1 = new VectorAssembler()
  .setInputCols(weatherFeatures1)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled1 = df_features1.select("features", "month")

val Array(trainingData1, testData1) = df_labeled1.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest1 = new RandomForestClassifier()
  .setLabelCol("month")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model1 = randomForest1.fit(trainingData1)

// predictions
val predictions1 = model1.transform(testData1)

// evaluate accuracy
val evaluator1 = new MulticlassClassificationEvaluator()
  .setLabelCol("month")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy1 = evaluator1.evaluate(predictions1)
println(s"Accuracy: $accuracy1")

// case2: Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input
val df_features2 = new VectorAssembler()
  .setInputCols(weatherFeatures2)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled2 = df_features2.select("features", "month")

val Array(trainingData2, testData2) = df_labeled2.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest2 = new RandomForestClassifier()
  .setLabelCol("month")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model2 = randomForest2.fit(trainingData2)

// predictions
val predictions2 = model2.transform(testData2)

// evaluate accuracy
val evaluator2 = new MulticlassClassificationEvaluator()
  .setLabelCol("month")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy2 = evaluator2.evaluate(predictions2)
println(s"Accuracy: $accuracy2")

// case3: Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input
val df_features3 = new VectorAssembler()
  .setInputCols(weatherFeatures3)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled3 = df_features3.select("features", "month")

val Array(trainingData3, testData3) = df_labeled3.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest3 = new RandomForestClassifier()
  .setLabelCol("month")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model3 = randomForest3.fit(trainingData3)

// predictions
val predictions3 = model3.transform(testData3)

// evaluate accuracy
val evaluator3 = new MulticlassClassificationEvaluator()
  .setLabelCol("month")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy3 = evaluator3.evaluate(predictions3)
println(s"Accuracy: $accuracy3")


// case4: Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
val df_features4 = new VectorAssembler()
  .setInputCols(weatherFeatures1)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled4 = df_features4.select("features", "hour")

val Array(trainingData4, testData4) = df_labeled4.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest4 = new RandomForestClassifier()
  .setLabelCol("hour")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model4 = randomForest4.fit(trainingData4)

// predictions
val predictions4 = model4.transform(testData4)

// evaluate accuracy
val evaluator4 = new MulticlassClassificationEvaluator()
  .setLabelCol("hour")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy4 = evaluator4.evaluate(predictions4)
println(s"Accuracy: $accuracy4")

// case5: Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
val df_features5 = new VectorAssembler()
  .setInputCols(weatherFeatures2)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled5 = df_features5.select("features", "hour")

val Array(trainingData5, testData5) = df_labeled5.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest5 = new RandomForestClassifier()
  .setLabelCol("hour")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model5 = randomForest5.fit(trainingData5)

// predictions
val predictions5 = model5.transform(testData5)

// evaluate accuracy
val evaluator5 = new MulticlassClassificationEvaluator()
  .setLabelCol("hour")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy5 = evaluator5.evaluate(predictions5)
println(s"Accuracy: $accuracy5")

// case 6: Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
val df_features6 = new VectorAssembler()
  .setInputCols(weatherFeatures3)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled6 = df_features6.select("features", "hour")

val Array(trainingData6, testData6) = df_labeled6.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest6 = new RandomForestClassifier()
  .setLabelCol("hour")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model6 = randomForest6.fit(trainingData6)

// predictions
val predictions6 = model6.transform(testData6)

// evaluate accuracy
val evaluator6 = new MulticlassClassificationEvaluator()
  .setLabelCol("hour")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy6 = evaluator6.evaluate(predictions6)
println(s"Accuracy: $accuracy6")


// Pt. 6 FOR MONTH

// additional metrics
def customEvaluatorMonth(predictions: DataFrame, labelCol: String, predictionCol: String, probabilityCol: String): Unit = {
  
  // 12 months --> starting over
  val cyclicDistance = (correct: Int, predicted: Int) => {
    val diff = math.abs(correct - predicted)
    math.min(diff, 12 - diff)
  }

  val enrichedPredictions = predictions.withColumn("correct",
    col(labelCol) === col(predictionCol))
    .withColumn("cyclic_distance",
      udf((correct: Double, predicted: Double) => cyclicDistance(correct.toInt, predicted.toInt))
        .apply(col(labelCol), col(predictionCol)))
    .withColumn("prob_correct",
      udf((probs: Vector, correct: Double) => probs(correct.toInt - 1))
        .apply(col(probabilityCol), col(labelCol)))

  val total = predictions.count().toDouble
  val correctCount = enrichedPredictions.filter(col("correct")).count()
  val oneAwayCount = enrichedPredictions.filter(col("cyclic_distance") <= 1).count()
  val twoAwayCount = enrichedPredictions.filter(col("cyclic_distance") <= 2).count()
  val avgCorrectProb = enrichedPredictions.agg(avg(col("prob_correct"))).head().getDouble(0)

  println(f"Custom Evaluation Metrics:")
  println(f"Accuracy (correct predictions): ${(correctCount / total) * 100}%.2f%%")
  println(f"Accuracy (within 1 of correct): ${(oneAwayCount / total) * 100}%.2f%%")
  println(f"Accuracy (within 2 of correct): ${(twoAwayCount / total) * 100}%.2f%%")
  println(f"Average probability for correct values: ${avgCorrectProb * 100}%.2f%%")
}

// apply custom evaluator for each case
println("Custom Evaluator Results for Case 1:")
customEvaluatorMonth(predictions1, "month", "prediction", "probability")

println("\nCustom Evaluator Results for Case 2:")
customEvaluatorMonth(predictions2, "month", "prediction", "probability")

println("\nCustom Evaluator Results for Case 3:")
customEvaluatorMonth(predictions3, "month", "prediction", "probability")


// Pt. 6 FOR HOUR

// additional metrics
def customEvaluatorHour(predictions: DataFrame, labelCol: String, predictionCol: String, probabilityCol: String): Unit = {
  
  // 24 hours --> starting over
  val cyclicDistance = (correct: Int, predicted: Int, cycleLength: Int) => {
    val diff = math.abs(correct - predicted)
    math.min(diff, cycleLength - diff)
  }

  val cycleLength = 24 

  val enrichedPredictions = predictions.withColumn("correct",
    col(labelCol) === col(predictionCol))
    .withColumn("cyclic_distance",
      udf((correct: Double, predicted: Double) => cyclicDistance(correct.toInt, predicted.toInt, cycleLength))
        .apply(col(labelCol), col(predictionCol)))
    .withColumn("prob_correct",
      udf((probs: Vector, correct: Double) => probs(correct.toInt))
        .apply(col(probabilityCol), col(labelCol))) // Corrected indexing logic

  val total = predictions.count().toDouble
  val correctCount = enrichedPredictions.filter(col("correct")).count()
  val oneAwayCount = enrichedPredictions.filter(col("cyclic_distance") <= 1).count()
  val twoAwayCount = enrichedPredictions.filter(col("cyclic_distance") <= 2).count()
  val avgCorrectProb = enrichedPredictions.agg(avg(col("prob_correct"))).head().getDouble(0)

  println(f"Custom Evaluation Metrics:")
  println(f"Accuracy (correct predictions): ${(correctCount / total) * 100}%.2f%%")
  println(f"Accuracy (within 1 of correct): ${(oneAwayCount / total) * 100}%.2f%%")
  println(f"Accuracy (within 2 of correct): ${(twoAwayCount / total) * 100}%.2f%%")
  println(f"Average probability for correct values: ${avgCorrectProb * 100}%.2f%%")
}

// apply custom evaluator for each case
println("Custom Evaluator Results for Case 4:")
customEvaluatorHour(predictions4, "hour", "prediction", "probability")

println("\nCustom Evaluator Results for Case 5:")
customEvaluatorHour(predictions5, "hour", "prediction", "probability")

println("\nCustom Evaluator Results for Case 6:")
customEvaluatorHour(predictions6, "hour", "prediction", "probability")


// Last part: implementing own things 

// Linear regression with case 1:
val linearRegression = new LinearRegression()
  .setLabelCol("month")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")

val lrModel = linearRegression.fit(trainingData1)

val predictionslr = lrModel.transform(testData1)

// rms
val evaluatorlr = new RegressionEvaluator()
  .setLabelCol("month")
  .setPredictionCol("prediction")
  .setMetricName("rmse") 

val rmselr = evaluatorlr.evaluate(predictionslr)
println(s"Root Mean Square Error (RMSE) on test data: $rmselr")

// acc
val roundedPredictions = predictions1.withColumn("rounded_prediction", round(col("prediction")))
val accuracylr = roundedPredictions.filter(col("rounded_prediction") === col("month")).count().toDouble / testData1.count()
println(f"Accuracy after rounding: ${accuracylr * 100}%.2f%%")


// case 7: Predict the month (1-12) using the price and humidity
val weatherFeatures4 = Array("humidity", "electricity_price") 

val df_features7 = new VectorAssembler()
  .setInputCols(weatherFeatures4)
  .setOutputCol("features")
  .transform(df_with_labels)

val df_labeled7 = df_features7.select("features", "month")

val Array(trainingData7, testData7) = df_labeled7.randomSplit(Array(0.8, 0.2), seed = 1)

val randomForest7 = new RandomForestClassifier()
  .setLabelCol("month")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

val model7 = randomForest7.fit(trainingData7)

// predictions
val predictions7 = model7.transform(testData7)

// evaluate accuracy
val evaluator7 = new MulticlassClassificationEvaluator()
  .setLabelCol("month")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy7 = evaluator7.evaluate(predictions7)
println(s"Accuracy: $accuracy7")