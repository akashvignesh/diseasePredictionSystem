FROM python:3.12.4

WORKDIR /DISEASEPREDICTIONSYSTEM

COPY requirements.txt .

# Copy the requirements file and install dependencies
RUN pip install  -r requirements.txt

# Run the application

CMD ["python", "/Application/back_end/main.py"]
