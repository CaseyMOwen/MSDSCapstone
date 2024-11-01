# Build and run at appropriate port, cleaning previous images:
# docker build --tag model-api-image:latest --label capstone-api ./API; docker system prune --force --filter='label=capstone-api'; PORT=8080 && docker run -p 9090:${PORT} -e PORT=${PORT} --rm --name model-api-container model-api-image

# Start docker desktop: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

# Python base image
FROM python:3.10-bookworm

# The /app directory should act as the main application directory
WORKDIR /app

# Copy ONLY requirements. This layer is cached if this file doesn't change
COPY requirements.txt . 

# This uses the cached layer if the previous layer was cached aswell
RUN pip install -r requirements.txt 

# Copy all directories to docker base image
COPY . .

# Python dependencies
# RUN pip install xgboost==2.1.0
# RUN pip install numpy==2.0.0
# RUN pip install pandas==2.2.2
# RUN pip install flask
# RUN pip install gunicorn
# RUN pip install flask-cors
# RUN pip install scikit-learn==1.5.1
# RUN pip install category_encoders
# RUN pip install pyarrow



ENV PORT 8080
# Make port 8080 available to the worl outside container
# EXPOSE 8080

# Start the app 
CMD exec gunicorn --bind :$PORT app:app
# CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]