FROM python:3.9-slim

#working directory
WORKDIR /app

#install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#Copy app code
COPY . .

#Expose port
EXPOSE 5000

#run the app
CMD ["python", "app.py"]