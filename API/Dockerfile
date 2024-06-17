# Python base image
FROM python:3.9-slim

# The /app directory should act as the main application directory
WORKDIR /app



# Python dependencies
RUN pip install xgboost
RUN pip install pandas
RUN pip install flask
RUN pip install gunicorn

# Copy all directories to docker base image
COPY . .

ENV PORT 8080
# Make port 8080 available to the worl outside container
# EXPOSE 8080

# Start the app 
CMD exec gunicorn --bind :$PORT app:app
# CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]