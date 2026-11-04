FROM jenkins/jenkins:lts-jdk21

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y docker-cli curl \
    && rm -rf /var/lib/apt/lists/*
RUN jenkins-plugin-cli --plugins workflow-aggregator git docker-workflow pipeline-stage-view

USER jenkins
