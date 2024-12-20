FROM python:3.12.4

# Set the working directory
WORKDIR /DISEASEPREDICTIONSYSTEM/Application/back_end

# Copy the necessary files
COPY ./Application /DISEASEPREDICTIONSYSTEM/Application
COPY ./Model/disease_prediction_model.pkl /DISEASEPREDICTIONSYSTEM/Model/disease_prediction_model_version_two.pkl
COPY ./Model/scaler.pkl /DISEASEPREDICTIONSYSTEM/Model/scaler.pkl

# Install dependencies
COPY requirements.txt /DISEASEPREDICTIONSYSTEM/requirements.txt
RUN pip install -r /DISEASEPREDICTIONSYSTEM/requirements.txt

# Run the application
CMD ["fastapi", "run", "main.py", "--port", "80"]
