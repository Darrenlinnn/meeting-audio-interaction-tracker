# Dashboard

```bash
pip install streamlit streamlit-autorefresh boto3 pandas altair
export S3_BUCKET=<your-bucket> AWS_REGION=us-east-1   # credentials via IAM role or env vars
streamlit run dashboard/app.py
```

The deployed version ran on an EC2 instance with an IAM role that had read access to the S3 bucket. Never put AWS keys in code.
