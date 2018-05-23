FROM bvlc/caffe:cpu
RUN pip install boto3
WORKDIR /workspace
COPY classifier.py .
COPY gender_model gender_model
COPY nsfw_model nsfw_model
EXPOSE 80
CMD ["python", "/workspace/classifier.py"]