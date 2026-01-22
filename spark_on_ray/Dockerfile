# Anyscale Container-Compatible Dockerfile
FROM anyscale/ray:2.53.0-slim-py312-cu128

# Environment variables
# ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/ray
ENV PATH=/home/ray/anaconda3/bin:$PATH

# Hadoop and Spark versions
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# PySpark uses py4j from site-packages - RayDP needs to find it
ENV SPARK_HOME=/home/ray/anaconda3/lib/python3.12/site-packages/pyspark

# Ray JAR path for Spark/RayDP JVM communication
ENV RAY_JARS_DIR=/home/ray/anaconda3/lib/python3.12/site-packages/ray/jars
ENV CLASSPATH=$RAY_JARS_DIR/*:$CLASSPATH

# Spark config directory
ENV SPARK_CONF_DIR=/home/ray/.spark

# Add Hadoop and Spark to PATH
ENV PATH=$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH

# Hadoop classpath for S3 access
ENV HADOOP_CLASSPATH=$HADOOP_HOME/share/hadoop/tools/lib/*:$HADOOP_CLASSPATH

# Install system dependencies including Hadoop requirements (using sudo since base image runs as ray user)
RUN sudo apt-get update -y && \
    sudo apt-get install -y --no-install-recommends \
    tzdata \
    openssh-client \
    openssh-server \
    rsync \
    zip \
    unzip \
    git \
    gdb \
    openjdk-11-jdk \
    maven \
    build-essential \
    gcc \
    g++ \
    python3-venv \
    python3-dev \
    curl \
    wget \
    libsnappy-dev \
    libssl-dev \
    liblz4-dev \
    zlib1g-dev && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo mkdir -p /var/run/sshd

# Download and install Hadoop
RUN sudo mkdir -p /opt && \
    cd /tmp && \
    wget -q https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    sudo tar -xzf hadoop-${HADOOP_VERSION}.tar.gz -C /opt && \
    sudo mv /opt/hadoop-${HADOOP_VERSION} /opt/hadoop && \
    rm hadoop-${HADOOP_VERSION}.tar.gz && \
    sudo mkdir -p $HADOOP_HOME/logs

# Download AWS SDK JARs for S3 access
RUN cd /tmp && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar && \
    sudo mv hadoop-aws-3.3.6.jar $HADOOP_HOME/share/hadoop/tools/lib/ && \
    sudo mv aws-java-sdk-bundle-1.12.367.jar $HADOOP_HOME/share/hadoop/tools/lib/

# Create Hadoop configuration directory and basic configs
RUN sudo mkdir -p $HADOOP_HOME/etc/hadoop && \
    echo '<?xml version="1.0"?>' | sudo tee $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '<configuration>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '  <property>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '    <name>fs.s3a.impl</name>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '    <value>org.apache.hadoop.fs.s3a.S3AFileSystem</value>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '  </property>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '  <property>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '    <name>fs.s3a.aws.credentials.provider</name>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '    <value>com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain</value>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '  </property>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null && \
    echo '</configuration>' | sudo tee -a $HADOOP_HOME/etc/hadoop/core-site.xml > /dev/null

# Set proper permissions for ray user
RUN sudo chown -R ray:users $HADOOP_HOME && \
    sudo chmod -R 755 $HADOOP_HOME

# Install Python packages to Anaconda environment (no virtualenv needed)
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir wheel

# Install core packages
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir anyscale jupyterlab

# Install PySpark with matching Hadoop version
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir pyspark==3.5.0

# Install py4j (Python-Java bridge required by RayDP)
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir py4j

# Install raydp
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir --pre raydp

# Extract ray_dist.jar from Ray wheel and backup for restoration after pre-start
# The Anyscale pre-start script replaces Ray at container startup, removing the JAR.
RUN RAY_VERSION=$(/home/ray/anaconda3/bin/python -c "import ray; print(ray.__version__)") && \
    cd /tmp && \
    /home/ray/anaconda3/bin/pip download "ray==${RAY_VERSION}" --no-deps -d . && \
    mkdir -p /home/ray/anaconda3/lib/python3.12/site-packages/ray/jars && \
    unzip -o -j ray-*.whl "ray/jars/*" -d /home/ray/anaconda3/lib/python3.12/site-packages/ray/jars/ && \
    rm -f ray-*.whl && \
    sudo mkdir -p /opt/ray-jars-backup && \
    sudo cp /home/ray/anaconda3/lib/python3.12/site-packages/ray/jars/ray_dist.jar /opt/ray-jars-backup/

# Patch pre-start script to restore ray_dist.jar after Ray replacement
RUN if [ -f /opt/anyscale/ray-prestart ]; then \
        echo '' | sudo tee -a /opt/anyscale/ray-prestart > /dev/null && \
        echo '# RAYDP FIX: Restore ray_dist.jar' | sudo tee -a /opt/anyscale/ray-prestart > /dev/null && \
        echo 'if [[ -f /opt/ray-jars-backup/ray_dist.jar ]]; then' | sudo tee -a /opt/anyscale/ray-prestart > /dev/null && \
        echo '    "${SUDO[@]}" mkdir -p "${ANYSCALE_RAY_SITE_PKG_DIR}/ray/jars"' | sudo tee -a /opt/anyscale/ray-prestart > /dev/null && \
        echo '    "${SUDO[@]}" cp /opt/ray-jars-backup/ray_dist.jar "${ANYSCALE_RAY_SITE_PKG_DIR}/ray/jars/"' | sudo tee -a /opt/anyscale/ray-prestart > /dev/null && \
        echo 'fi' | sudo tee -a /opt/anyscale/ray-prestart > /dev/null; \
    fi

# Install additional Python packages for Spark/Hadoop integration
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir emoji pyarrow pandas numpy findspark

# Configure bash environment (minimal - for Anyscale workspace compatibility only)
RUN echo 'PROMPT_COMMAND="history -a"' >> /home/ray/.bashrc && \
    echo '[ -e ~/.workspacerc ] && source ~/.workspacerc' >> /home/ray/.bashrc

# Create Spark configuration for S3 access and Ray JAR classpath
RUN mkdir -p /home/ray/.spark && \
    echo 'spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem' > /home/ray/.spark/spark-defaults.conf && \
    echo 'spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain' >> /home/ray/.spark/spark-defaults.conf && \
    echo 'spark.jars.packages=org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.367' >> /home/ray/.spark/spark-defaults.conf && \
    echo "spark.driver.extraClassPath=/home/ray/anaconda3/lib/python3.12/site-packages/ray/jars/*" >> /home/ray/.spark/spark-defaults.conf && \
    echo "spark.executor.extraClassPath=/home/ray/anaconda3/lib/python3.12/site-packages/ray/jars/*" >> /home/ray/.spark/spark-defaults.conf

# Verify Python packages and Ray Java JAR
RUN /home/ray/anaconda3/bin/python -c "import ray; print(f'Ray version: {ray.__version__}')" && \
    /home/ray/anaconda3/bin/python -c "import py4j; print(f'py4j version: {py4j.__version__}')" && \
    /home/ray/anaconda3/bin/python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')" && \
    /home/ray/anaconda3/bin/python -c "import raydp; print('raydp installed successfully')" && \
    /home/ray/anaconda3/bin/python -c "import pyarrow; print('PyArrow installed successfully')" && \
    /home/ray/anaconda3/bin/python -c "import ray, os; jar_path = os.path.join(os.path.dirname(ray.__file__), 'jars', 'ray_dist.jar'); assert os.path.exists(jar_path), f'ray_dist.jar not found at {jar_path}'; print(f'ray_dist.jar found: {jar_path}')"

# Set working directory
WORKDIR /home/ray
