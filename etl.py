import configparser
from datetime import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, year, month, dayofweek, dayofmonth, hour, weekofyear, monotonically_increasing_id


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session() -> SparkSession:
    spark = SparkSession.builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0").getOrCreate()
    return spark


def process_song_data(spark: SparkSession, input_data: str, output_data: str) -> None:
    song_data = input_data + 'song_data/*/*/*/*.json'
    df = spark.read.json(song_data)

    # Songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration').dropDuplicates()
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs/songs.parquet'), 'overwrite')
    print("Songs table created")

    # Artists table
    artists_table = df.select('artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude') \
                      .withColumnRenamed('artist_name', 'name') \
                      .withColumnRenamed('artist_location', 'location') \
                      .withColumnRenamed('artist_latitude', 'latitude') \
                      .withColumnRenamed('artist_longitude', 'longitude') \
                      .dropDuplicates()
    artists_table.write.parquet(os.path.join(output_data, 'artists/artists.parquet'), 'overwrite')
    print("Artists table created")


def process_log_data(spark: SparkSession, input_data: str, output_data: str) -> None:
    log_data = input_data + 'log_data/*/*/*.json'
    df = spark.read.json(log_data)
    df = df.filter(df.page == 'NextSong')

    # USERS table
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level').dropDuplicates()
    users_table.write.parquet(os.path.join(output_data, 'users/users.parquet'), 'overwrite')
    print("Users table created")

    # TIME table
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    df = df.withColumn('datetime', get_datetime(df.ts))
    time_table = df.select('datetime') \
                   .withColumn('start_time', df.datetime) \
                   .withColumn('hour', hour('datetime')) \
                   .withColumn('day', dayofmonth('datetime')) \
                   .withColumn('week', weekofyear('datetime')) \
                   .withColumn('month', month('datetime')) \
                   .withColumn('year', year('datetime')) \
                   .withColumn('weekday', dayofweek('datetime')) \
                   .dropDuplicates()
    time_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'time/time.parquet'), 'overwrite')
    print("Time table created")

    # SONGPLAYS table
    song_df = spark.read.json(input_data + 'song_data/A/A/A/*.json')
    joined_df = df.join(song_df, df['artist'] == song_df['artist_name'], 'inner')
    songplays_table = joined_df.select(
        col('datetime').alias('start_time'),
        col('userId').alias('user_id'),
        col('level').alias('level'),
        col('song_id').alias('song_id'),
        col('artist_id').alias('artist_id'),
        col('sessionId').alias('session_id'),
        col('location').alias('location'),
        col('userAgent').alias('user_agent'),
        year('datetime').alias('year'),
        month('datetime').alias('month')) \
        .withColumn('songplay_id', monotonically_increasing_id())
    songplays_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'songplays/songplays.parquet'), 'overwrite')
    print("Songplays table created")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://datalake.ram/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
