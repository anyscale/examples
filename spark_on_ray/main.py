from pyspark.sql.types import StructType, StructField, FloatType, StringType
import os
import urllib.request
import ray
import raydp

# Fetch a small dataset
path = "/mnt/cluster_storage/iris.csv"
if not os.path.exists(path):
    urllib.request.urlretrieve(
        "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv",
        path,
    )

ray.init()

num_executors = 128
executor_memory = "3GB"

# Get the Ray JAR path for Spark classpath
ray_jars_dir = os.path.join(os.path.dirname(ray.__file__), "jars")
ray_dist_jar = os.path.join(ray_jars_dir, "ray_dist.jar")

spark = raydp.init_spark(
    app_name="RayDP Example",
    num_executors=num_executors,
    executor_cores=1,
    executor_memory=executor_memory,
    configs={
        "spark.jars": ray_dist_jar,
        "spark.driver.extraClassPath": ray_jars_dir + "/*",
        "spark.executor.extraClassPath": ray_jars_dir + "/*",
    }
)


def main():
    # Define a schema for the Iris dataset (assuming the CSV includes a header)
    iris_schema = StructType([
        StructField("sepal_length", FloatType(), True),
        StructField("sepal_width", FloatType(), True),
        StructField("petal_length", FloatType(), True),
        StructField("petal_width", FloatType(), True),
        StructField("species", StringType(), True)
    ])

    # Read the Iris dataset from CSV into a DataFrame
    iris_df = spark.read.csv(
        "/mnt/cluster_storage/iris.csv",
        schema=iris_schema,
        header=True,
        inferSchema=False
    )

    # Display the first few rows
    print("Sample rows from the Iris dataset:")
    iris_df.show(5)

    # Print the schema of the DataFrame
    print("Schema of the Iris DataFrame:")
    iris_df.printSchema()

    # Generate summary statistics
    print("Basic statistics (describe):")
    iris_df.describe().show()

    # Group rows by 'species' and count them
    print("Count each species:")
    iris_df.groupBy("species").count().show()

    # Create a new column (example transformation)
    iris_df = iris_df.withColumn("sepal_ratio", iris_df["sepal_length"] / iris_df["sepal_width"])
    print("DataFrame with new 'sepal_ratio' column:")
    iris_df.show(5)

    # Filter rows to show only those matching a certain condition (e.g., wide sepals)
    filtered_df = iris_df.filter(iris_df.sepal_width > 3.5)
    print("Rows with sepal_width > 3.5:")
    filtered_df.show()

    spark.stop()


if __name__ == "__main__":
    main()
