# Data Lake with Spark
This repository is for project 3 of the [Udacity Data Engineer Nanodegree](https://www.udacity.com/course/data-engineer-nanodegree--nd027) program. For this project, we are working with a (fictitious) music company called Sparkify -- a music streaming startup. We are tasked with building an ETL pipeline using Spark, which reads data from an `S3` bucket in `json` format (both archives of songs and song listening log data) and transforms it into a relational database (called "data lake" for the purposes of the exercise), finally saving the results as `parquet` files also in S3.

## How to run
### Prerequisites
In order to be able to run this project, you need the following:
1. AWS credentials (IAM user credentials, stored in the `dl.cfg` file; make sure you don't use any quotation marks)
2. Python 3.7 (and an environment with `pyspark` installed)
3. An `S3` bucket where you can store the parquet files to (specified as `output_data` in `main()` in `etl.py`)

### Run the script
To actually run the ETL process, all you need to do it run
```bash
python3 etl.py
```
in a bash terminal.

This will run the script that does the following:
1. Creates a `SparkSession`
2. Reads the `song_data` json files from Udacity's S3 bucket (`s3://udacity-dend/song_data`) and processes them into a Spark dataframe. Then:
   1. Processes the song data into two tables: `songs` and `artists`. These constitute the fact tables that contain all information about possible sonds and artists (the source is from the [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/))
   2. Partitions the `songs` table by `year` and `artist`.
   3. Write the `songs` and `artists` tables as parquet files to an output bucket on S3 (private bucket, you must have permissions to write to it)
3. Reads the `log_data` json files from Udacity's S3 bucket (`s3://udacity-dend/log_data`) and processes them to a Spark dataframe. Then:
   1. Keeps only records of songs that were started (`page = NextSong`)
   2. Process any user data into a `users` Spark dataframe
   3. Process the unique timestamps (after transformation) into a `time` Spark dataframe
   4. Read again the `song_data` json files, and join it with the `log_data` on song name and song duration. From here, create a `songplays` dataframe that is essentially a fact table for all the songs that were played on Sparkify.
   5. Finally, write all dataframes to S3 as `parquet` files (the `time` and `songplays` tables are partitioned by `year` and `month`)


