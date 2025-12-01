# Anyscale Container-Compatible Dockerfile
FROM anyscale/ray:2.50.0-slim-py312-cu128

# Environment variables
ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/ray
ENV PATH=/home/ray/anaconda3/bin:$PATH

# Hadoop and Spark versions
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# PySpark uses py4j from site-packages - RayDP needs to find it
ENV SPARK_HOME=/home/ray/anaconda3/lib/python3.12/site-packages/pyspark

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

# Install additional Python packages for Spark/Hadoop integration
RUN /home/ray/anaconda3/bin/pip install --no-cache-dir emoji pyarrow pandas numpy findspark

# Configure bash environment (minimal - for Anyscale workspace compatibility only)
RUN echo 'PROMPT_COMMAND="history -a"' >> /home/ray/.bashrc && \
    echo '[ -e ~/.workspacerc ] && source ~/.workspacerc' >> /home/ray/.bashrc

# Create Spark configuration for S3 access
RUN mkdir -p /home/ray/.spark && \
    echo 'spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem' > /home/ray/.spark/spark-defaults.conf && \
    echo 'spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain' >> /home/ray/.spark/spark-defaults.conf && \
    echo 'spark.jars.packages=org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.367' >> /home/ray/.spark/spark-defaults.conf

# Verify Python packages only (Hadoop/Java verification can cause container startup issues)
RUN /home/ray/anaconda3/bin/python -c "import ray; print(f'Ray version: {ray.__version__}')" && \
    /home/ray/anaconda3/bin/python -c "import py4j; print(f'py4j version: {py4j.__version__}')" && \
    /home/ray/anaconda3/bin/python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')" && \
    /home/ray/anaconda3/bin/python -c "import raydp; print('raydp installed successfully')" && \
    /home/ray/anaconda3/bin/python -c "import pyarrow; print('PyArrow installed successfully')"

# Set working directory
WORKDIR /home/ray
