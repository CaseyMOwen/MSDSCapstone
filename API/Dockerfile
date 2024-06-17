# Start your image with a node base image
FROM python:3.9-slim

# The /app directory should act as the main application directory
WORKDIR /app

# Copy the app package and package-lock.json file

RUN pip install xgboost
RUN pip install pandas
RUN pip install flask
RUN pip install gunicorn

# Copy local directories to the current local directory of our docker image (/app)
COPY . .


EXPOSE 5000

# Start the app using serve command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]